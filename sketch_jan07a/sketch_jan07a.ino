#include <ESP8266WiFi.h>
#include <HX711_ADC.h>
#include <EEPROM.h>

//pins for Load Cell
const int HX711_dout = D0; //mcu > HX711 dout pin
const int HX711_sck = D1; //mcu > HX711 sck pin
#define triggerPin  D3
#define echoPin     D2

//HX711 constructor for Load Cell:
HX711_ADC LoadCell(HX711_dout, HX711_sck);

// WiFi parameters to be configured
const char *ssid = "CALLISTA";
const char *password = "semangatbaru";
const char *host = "https://ajinaufal.website/";

//milis 
unsigned long previousMillis=0; // millis() returns an unsigned long.

void setup() {
  Serial.begin(9600);
  Serial.println("Starting...");
  // Wifi Setup
  WiFi.mode(WIFI_OFF);        //Prevents reconnection issue (taking too long to connect)
  delay(200);
  WiFi.mode(WIFI_STA);        //This line hides the viewing of ESP as wifi hotspot
  WiFi.begin(ssid, password); // Connect to WiFi
  Serial.print("Connecting"); // Wait for connection
  while (WiFi.status() != WL_CONNECTED) { // then after it connected, get out of the loop
     delay(500);
     Serial.print(".");// while wifi not connected yet, print '.'
  }
  Serial.println(""); //print a new line, then print WiFi connected and the IP address
  Serial.println("WiFi connected");
  Serial.println(WiFi.localIP());// Print the IP address
  Serial.println("starts a sensor reading");

  // Ultrasonic setup
  pinMode(triggerPin, OUTPUT);
  pinMode(echoPin, INPUT);
  
  // Load cell setup
  LoadCell.begin();
  float calibrationValue; // calibration value (see example file "Calibration.ino")
  calibrationValue = 696.0; // uncomment this if you want to set the calibration value in the sketch

  unsigned long stabilizingtime = 2000; // preciscion right after power-up can be improved by adding a few seconds of stabilizing time
  boolean _tare = true; //set this to false if you don't want tare to be performed in the next step
  LoadCell.start(stabilizingtime, _tare);
  if (LoadCell.getTareTimeoutFlag()) {
    Serial.println("Timeout, check MCU>HX711 wiring and pin designations");
    while (1);
  }
  else {
    LoadCell.setCalFactor(calibrationValue); // set calibration value (float)
    Serial.println("Startup is complete");
  }
}

void loop() {
  // Wifi loop
  HTTPClient http;    //Declare object of class HTTPClient
  http.begin("https://ajinaufal.website/api/scales"); //Specify request destination
  http.addHeader("Content-Type", "application/x-www-form-urlencoded"); //Specify content-type header
  
  // Ultrasonic loop
  long duration, jarak;
  digitalWrite(triggerPin, LOW);
  delayMicroseconds(2); 
  digitalWrite(triggerPin, HIGH);
  delayMicroseconds(10); 
  digitalWrite(triggerPin, LOW);
  duration = pulseIn(echoPin, HIGH);
  jarak = (duration/2) / 29.1;
  
  // Load cell loop
  static boolean newDataReady = 0;
  if (LoadCell.update()) newDataReady = true; // check for new data/start next conversion
    if (newDataReady) { // get smoothed value from the dataset:
      if ((unsigned long)(currentMillis - previousMillis) >= 5000){
          int httpCode = http.POST("weight" + String(i) + "height" + String(jarak));
          http.end();  //Close connection
          newDataReady = 0; 
          previousMillis = millis();
        }else if ((unsigned long)(currentMillis - previousMillis) >= 100){
          float i = LoadCell.getData();
          Serial.print("Load_cell output val: ");
          Serial.print(i);
          Serial.print("Gram");
          Serial.print("      ");
          Serial.print("jarak :");
          Serial.print(jarak);
          Serial.println(" cm");
          newDataReady = 0; 
          previousMillis = millis();
          }
    }

  // receive command from serial terminal, send 't' to initiate tare operation:
  if (Serial.available() > 0) {
    char inByte = Serial.read();
    if (inByte == 't') LoadCell.tareNoDelay();
  }

  // check if last tare operation is complete:
  if (LoadCell.getTareStatus() == true) {
    Serial.println("Tare complete");
  }
}
