#include "secret.h"
#include "M5TimerCAM.h"
#include <WiFi.h>
#include <PubSubClient.h>

#define CAM_EXT_WAKEUP_PIN 4

//Objects
//WiFiClientSecure askClient; //SSL Client
WiFiClient askClient; //Non-SSL Client, also remove the comments for askClient.setCACert(local_root_ca);

PubSubClient client(askClient);
long lastMsg = 0;
char msg[50];
int value = 0;

static void jpegStream(WiFiClient* client);

void setup_wifi() {

  delay(10);
  // We start by connecting to a WiFi network
  WiFi.mode(WIFI_STA);
  WiFi.begin(ssid, password);
  WiFi.setSleep(false);

  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }

  randomSeed(micros());

  Serial.println("");
  Serial.println("WiFi connected");
  Serial.println("IP address: ");
  Serial.println(WiFi.localIP());
}

//MQTT callback
void callback(char* topic, byte * payload, unsigned int length) {

  for (int i = 0; i < length; i++) {
    Serial.println(topic);
    Serial.print(" has send ");
    Serial.print((char)payload[i]);
  }

}

//MQTT reconnect
void reconnect() {
  // Loop until we're reconnected
  while (!client.connected()) {
    Serial.print("********** Attempting MQTT connection...");
    // Attempt to connect
    if (client.connect(ClientID, username, mqttpass, lastwill, 1, 1, lastwillmsg)) {
      Serial.println("-> MQTT client connected");
      client.subscribe(topic);
      Serial.print("Subscribed to: ");
      Serial.println(topic);
    } else {
      Serial.print("failed, rc=");
      Serial.print(client.state());
      Serial.println("-> try again in 5 seconds");
      // Wait 5 seconds before retrying
      delay(5000);
    }
  }
}

void setup() {
    Serial.println("Wake up!!!");
    TimerCAM.begin();

    if (!TimerCAM.Camera.begin()) {
        Serial.println("Camera Init Fail");
        return;
    }

    Serial.println("Camera Init Success");

    TimerCAM.Camera.sensor->set_pixformat(TimerCAM.Camera.sensor,
                                          PIXFORMAT_JPEG);
    TimerCAM.Camera.sensor->set_framesize(TimerCAM.Camera.sensor,
                                          FRAMESIZE_SVGA);
    TimerCAM.Camera.sensor->set_vflip(TimerCAM.Camera.sensor, 1);
    TimerCAM.Camera.sensor->set_hmirror(TimerCAM.Camera.sensor, 0);

    setup_wifi();


    // askClient.setCACert(local_root_ca); //If you use non SSL then comment out
    client.setServer(mqtt_server, mqtt_port);
    client.setCallback(callback);

    jpegStream(&client);
    // sleep after 5s wakeup!
    // TimerCAM.Power.timerSleep(5);
    gpio_hold_en((gpio_num_t)POWER_HOLD_PIN);
    gpio_deep_sleep_hold_en();
    esp_sleep_enable_ext0_wakeup((gpio_num_t)CAM_EXT_WAKEUP_PIN,
                                 1);  // 1 = High, 0 = Low

    while (digitalRead(CAM_EXT_WAKEUP_PIN) == HIGH) {
        // wait for singal to go low
        delay(1);
    }

    // Go to sleep now
    Serial.println("Going to sleep now");
    esp_deep_sleep_start();
}

void loop() {

}

// used to image stream
#define PART_BOUNDARY "123456789000000000000987654321"
static const char* _STREAM_CONTENT_TYPE =
    "multipart/x-mixed-replace;boundary=" PART_BOUNDARY;
static const char* _STREAM_BOUNDARY = "\r\n--" PART_BOUNDARY "\r\n";
static const char* _STREAM_PART =
    "Content-Type: image/jpeg\r\nContent-Length: %u\r\n\r\n";

static void jpegStream(PubSubClient* client) {
    Serial.println("Image stream satrt");
    client->println("HTTP/1.1 200 OK");
    client->printf("Content-Type: %s\r\n", _STREAM_CONTENT_TYPE);
    client->println("Content-Disposition: inline; filename=capture.jpg");
    client->println("Access-Control-Allow-Origin: *");
    client->println();

    if (TimerCAM.Camera.get()) {
            TimerCAM.Power.setLed(255);
            Serial.printf("pic size: %d\n", TimerCAM.Camera.fb->len);

            client->print(_STREAM_BOUNDARY);
            client->printf(_STREAM_PART, TimerCAM.Camera.fb);
            int32_t to_sends    = TimerCAM.Camera.fb->len;
            int32_t now_sends   = 0;
            String out_buf    = (char*)(TimerCAM.Camera.fb->buf);
            uint32_t packet_len = 8 * 1024;
            uint32_t index = 0;
            while (to_sends > 0) {
                //To publish Strings:
                now_sends = to_sends > packet_len ? packet_len : to_sends;
                char img_str[now_sends];
                out_buf.toCharArray(img_str, now_sends, index);
                client->publish("yourrealm/ClientID/writeattributevalue/AttributeName/AssetID", img_str);
                index += now_sends;
                to_sends -= packet_len;
            }

            TimerCAM.Camera.free();
            TimerCAM.Power.setLed(0);
    }

client_exit:
    TimerCAM.Camera.free();
    TimerCAM.Power.setLed(0);

    Serial.printf("Image stream end\r\n");
}