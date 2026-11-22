// Octopus 16 firmware, OctopusMusic variant: EEG over BLE *and* USB serial.
//
// Derived from pieeg-club/Octopus_16 Framework/XIAO_ESP_32_S6.ino (ADC setup,
// pins, packet format, BLE UUIDs and register bridge) plus behaviour read off
// the shipped octoBridge build v1.0.0 (name "Octopus-16-XXXX", 8 MHz CLKIN on
// D6, heartbeat line). The gamepad / on-board DSP code is dropped.
//
// WHAT CHANGED vs the official firmware
//   * USB: send 'S' on the serial port and every sample is written as the same
//     51-byte packet the BLE characteristic carries (A0, counter, 16 x int24 BE,
//     C0). 'X' stops. While not streaming, a text heartbeat prints once a second.
//     12.75 kB/s over the ESP32-S3's built-in USB serial: no loss, no pairing.
//   * BLE: up to (MTU-3)/51 packets are batched into one notification (4 at
//     the usual MTU 256). The official build sends one notification per sample,
//     250/s, which macOS drops about half of. Parsers that split notifications
//     into 51-byte packets (the official quickstart, octomusic) are unaffected.
//
// SAFETY: the board has no isolation. Over USB, the computer must be on battery
// (charger unplugged, nothing mains-powered attached) while the headset is worn.
//
// Build: arduino-cli compile --fqbn esp32:esp32:XIAO_ESP32S3:USBMode=hwcdc,CDCOnBoot=default

#include <Arduino.h>
#include <SPI.h>
#include <esp_mac.h>
#include <esp_system.h>
#include <NimBLEDevice.h>
#include "ADS131M08.h"

#define FW_TAG "octomusic-usb-ble 1.1"

#define SERVICE_UUID   "4fafc201-1fb5-459e-8fcc-c5c9c331914b"
#define DATA_CHAR_UUID "beb5483e-36e1-4688-b7f5-ea07361b26a8"
#define CMD_CHAR_UUID  "c0de0001-36e1-4688-b7f5-ea07361b26a8"

#define PIN_CS1    D0
#define PIN_DRDY1  D1
#define PIN_RESET  D2
#define PIN_CS2    D3
#define PIN_CLKOUT D6
#define PIN_SCLK   D8
#define PIN_MISO   D9
#define PIN_MOSI   D10

#define CLKIN_HZ   8000000      // what the shipped build drives on D6
#define PKT        51
#define MAX_BATCH  4

ADS131M08 adc1;
ADS131M08 adc2;

NimBLEServer* pServer = nullptr;
NimBLECharacteristic* pData = nullptr;
NimBLECharacteristic* pCmd = nullptr;

TaskHandle_t adcTask = nullptr;
volatile bool usbStreaming = false;
volatile bool bleSubscribed = false;
volatile uint8_t bleBatch = 1;      // packets per notification, from the MTU
volatile uint32_t drdyCount = 0, usbDrops = 0;
volatile int activeChannels = 16;
uint8_t sampleCounter = 0;
char devName[24];
const char* resetReason = "?";

const char* resetReasonStr(esp_reset_reason_t r) {
  switch (r) {
    case ESP_RST_POWERON:   return "POWERON";
    case ESP_RST_EXT:       return "EXT_PIN";
    case ESP_RST_SW:        return "SOFTWARE";
    case ESP_RST_PANIC:     return "PANIC(crash)";
    case ESP_RST_INT_WDT:   return "INT_WATCHDOG";
    case ESP_RST_TASK_WDT:  return "TASK_WATCHDOG";
    case ESP_RST_WDT:       return "OTHER_WATCHDOG";
    case ESP_RST_DEEPSLEEP: return "DEEPSLEEP";
    case ESP_RST_BROWNOUT:  return "BROWNOUT(power dip)";
    case ESP_RST_SDIO:      return "SDIO";
    case ESP_RST_USB:       return "USB(host reset)";
    case ESP_RST_JTAG:      return "JTAG";
    default:                return "UNKNOWN";
  }
}

// register bridge (same protocol as the official firmware)
volatile bool has_pending_read = false, has_pending_write = false, use_mask = false;
volatile uint8_t reg_addr = 0;
volatile uint16_t reg_val = 0, reg_mask = 0;

void IRAM_ATTR onDrdy() {
  BaseType_t woken = pdFALSE;
  if (adcTask) vTaskNotifyGiveFromISR(adcTask, &woken);
  if (woken) portYIELD_FROM_ISR();
}

class ServerCB : public NimBLEServerCallbacks {
  void onConnect(NimBLEServer* s, NimBLEConnInfo& info) override {
    // 15-30 ms: what macOS/iOS accept. With batching that is plenty.
    s->updateConnParams(info.getConnHandle(), 12, 24, 0, 400);
  }
  void onDisconnect(NimBLEServer* s, NimBLEConnInfo& info, int reason) override {
    bleSubscribed = false;
    NimBLEDevice::getAdvertising()->start();
  }
  void onMTUChange(uint16_t mtu, NimBLEConnInfo& info) override {
    int b = (int(mtu) - 3) / PKT;
    bleBatch = (uint8_t)constrain(b, 1, MAX_BATCH);
  }
};

class DataCB : public NimBLECharacteristicCallbacks {
  void onSubscribe(NimBLECharacteristic* c, NimBLEConnInfo& info, uint16_t sub) override {
    bleSubscribed = sub != 0;
    int b = (int(info.getMTU()) - 3) / PKT;
    bleBatch = (uint8_t)constrain(b, 1, MAX_BATCH);
  }
};

class CmdCB : public NimBLECharacteristicCallbacks {
  void onWrite(NimBLECharacteristic* c, NimBLEConnInfo& info) override {
    std::string rx = c->getValue();
    if (rx.length() == 1) { reg_addr = rx[0]; has_pending_read = true; }
    else if (rx.length() == 3) {
      reg_addr = rx[0]; reg_val = (uint8_t(rx[1]) << 8) | uint8_t(rx[2]);
      use_mask = false; has_pending_write = true;
    } else if (rx.length() == 5) {
      reg_addr = rx[0]; reg_val = (uint8_t(rx[1]) << 8) | uint8_t(rx[2]);
      reg_mask = (uint8_t(rx[3]) << 8) | uint8_t(rx[4]);
      use_mask = true; has_pending_write = true;
    }
  }
};

static inline void put24(uint8_t* p, int32_t v) {
  p[0] = (v >> 16) & 0xFF; p[1] = (v >> 8) & 0xFF; p[2] = v & 0xFF;
}

void adcAcquisitionTask(void*) {
  static uint8_t batch[PKT * MAX_BATCH];
  uint8_t inBatch = 0;
  while (true) {
    ulTaskNotifyTake(pdTRUE, portMAX_DELAY);
    AdcOutput r1 = adc1.readAdcRaw();
    AdcOutput r2 = adc2.readAdcRaw();
    drdyCount++;

    uint8_t pkt[PKT];
    pkt[0] = 0xA0;
    pkt[1] = sampleCounter++;
    for (int i = 0; i < 8; i++) put24(&pkt[2 + i * 3], r1.ch[i].i);
    for (int i = 0; i < 8; i++) put24(&pkt[26 + i * 3], activeChannels == 16 ? r2.ch[i].i : 0);
    pkt[50] = 0xC0;

    if (usbStreaming) {
      if (Serial.availableForWrite() >= PKT) Serial.write(pkt, PKT);
      else usbDrops++;                       // host not reading: never block
    }

    if (bleSubscribed && pServer && pServer->getConnectedCount() > 0) {
      memcpy(&batch[inBatch * PKT], pkt, PKT);
      if (++inBatch >= bleBatch) {
        pData->notify(batch, inBatch * PKT);
        inBatch = 0;
      }
    } else {
      inBatch = 0;
    }
  }
}

void startClock() {
  // ADS CLKIN from D6, as the shipped build does. 80 MHz APB / 10 = exact.
  ledcAttach(PIN_CLKOUT, CLKIN_HZ, 1);
  ledcWrite(PIN_CLKOUT, 1);                 // 50% duty at 1-bit resolution
}

void setupAdc(ADS131M08& a) {
  a.setOsr(OSR_16384);
  a.writeRegisterMasked(0x08, 0x0F, 0x000F);   // DC block > 1 Hz
  a.writeRegister(REG_GAIN1, 0x2222);          // PGA x4, ch0-3
  a.writeRegister(REG_GAIN2, 0x2222);          // PGA x4, ch4-7
}

void setup() {
  Serial.setTxTimeoutMs(0);
  Serial.setRxBufferSize(256);
  Serial.begin(115200);
  delay(50);
  resetReason = resetReasonStr(esp_reset_reason());
  Serial.println("[BOOT] Octopus 16 EEG firmware starting... (" FW_TAG ")");
  Serial.printf("[BOOT] reset reason: %s\n", resetReason);

  startClock();
  Serial.printf("[BOOT] ADS CLKIN on D6 (%d Hz)\n", CLKIN_HZ);

  SPI.begin(PIN_SCLK, PIN_MISO, PIN_MOSI, -1);
  adc1.begin(PIN_SCLK, PIN_MISO, PIN_MOSI, PIN_CS1, PIN_DRDY1, PIN_RESET);
  adc2.begin(PIN_SCLK, PIN_MISO, PIN_MOSI, PIN_CS2, -1, PIN_RESET);
  adc1.reset();                                // shared reset line: both chips
  setupAdc(adc1);
  setupAdc(adc2);

  uint16_t g2 = adc2.readRegister(REG_GAIN1);
  activeChannels = (g2 == 0x0000 || g2 == 0xFFFF) ? 8 : 16;
  Serial.printf("[SYSTEM] ADC2 %s. %d-channel mode.\n",
                activeChannels == 16 ? "found" : "not found", activeChannels);

  uint8_t mac[6] = {0};
  if (esp_read_mac(mac, ESP_MAC_BT) != ESP_OK) esp_efuse_mac_get_default(mac);
  snprintf(devName, sizeof(devName), "Octopus-16-%02X%02X", mac[4], mac[5]);

  NimBLEDevice::init(devName);
  NimBLEDevice::setPower(ESP_PWR_LVL_P3);
  NimBLEDevice::setMTU(256);
  pServer = NimBLEDevice::createServer();
  pServer->setCallbacks(new ServerCB());
  NimBLEService* svc = pServer->createService(SERVICE_UUID);
  pData = svc->createCharacteristic(DATA_CHAR_UUID, NIMBLE_PROPERTY::NOTIFY);
  pData->setCallbacks(new DataCB());
  pCmd = svc->createCharacteristic(CMD_CHAR_UUID, NIMBLE_PROPERTY::WRITE | NIMBLE_PROPERTY::NOTIFY);
  pCmd->setCallbacks(new CmdCB());
  svc->start();

  NimBLEAdvertising* adv = NimBLEDevice::getAdvertising();
  NimBLEAdvertisementData primary;
  primary.setFlags(0x06);
  primary.setCompleteServices(NimBLEUUID(SERVICE_UUID));
  adv->setAdvertisementData(primary);
  NimBLEAdvertisementData scan;
  scan.setName(devName);
  adv->setScanResponseData(scan);
  adv->enableScanResponse(true);
  adv->start();
  Serial.printf("[BOOT] EEG advertising active as \"%s\".\n", devName);

  xTaskCreatePinnedToCore(adcAcquisitionTask, "ADCTask", 8192, nullptr, 10, &adcTask, 1);
  pinMode(PIN_DRDY1, INPUT_PULLUP);
  attachInterrupt(digitalPinToInterrupt(PIN_DRDY1), onDrdy, FALLING);
  Serial.println("[BOOT] Setup complete. Send 'S' to stream binary over USB, 'X' to stop.");
}

void loop() {
  while (Serial.available()) {
    int c = Serial.read();
    if (c == 'S') usbStreaming = true;
    else if (c == 'X') usbStreaming = false;
    else if (c == 'I' && !usbStreaming)
      Serial.printf("[INFO] %s name=%s ch=%d\n", FW_TAG, devName, activeChannels);
  }

  static uint32_t lastHb = millis(), lastCount = 0;
  if (millis() - lastHb >= 1000) {
    uint32_t n = drdyCount;
    // every second when idle, every 5 s while streaming (text lines between
    // packets are fine: the host splits on the A0..C0 framing)
    static uint8_t tick = 0;
    if (!usbStreaming || ++tick >= 5) {
      tick = 0;
      Serial.printf("[HB] up=%lus name=%s conn=%d sub=%d batch=%d ch=%d drdy=%lu rate=%lu/s usbdrop=%lu rst=%s heap=%lu\n",
                    millis() / 1000, devName, pServer ? pServer->getConnectedCount() : 0,
                    (int)bleSubscribed, (int)bleBatch, activeChannels, n, n - lastCount, usbDrops,
                    resetReason, (unsigned long)ESP.getFreeHeap());
    }
    lastCount = n;
    lastHb = millis();
  }

  if (has_pending_read) {
    uint16_t v = adc1.readRegister(reg_addr);
    uint8_t tx[3] = {reg_addr, (uint8_t)(v >> 8), (uint8_t)(v & 0xFF)};
    pCmd->notify(tx, 3);
    has_pending_read = false;
  }
  if (has_pending_write) {
    if (use_mask) {
      adc1.writeRegisterMasked(reg_addr, reg_val, reg_mask);
      adc2.writeRegisterMasked(reg_addr, reg_val, reg_mask);
    } else {
      adc1.writeRegister(reg_addr, reg_val);
      adc2.writeRegister(reg_addr, reg_val);
    }
    has_pending_write = false;
  }
  delay(5);
}
