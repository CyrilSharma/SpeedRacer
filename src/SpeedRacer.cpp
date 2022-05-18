#include <Arduino.h>
#include "rpLidar.h"
#include "rpLidarTypes.h"
#include <esp_task_wdt.h>
#include <vector>
#include <Constants.h>
#include <Servo.h>

Servo steeringServo;
Servo throttleServo;
rpLidar lidar(&Serial2,115200,13,12);

double getAverageDistance(const std::vector<scanDot> dots, float min_angle, float max_angle)
{
  float distanceSum = 0;
  int n = 0;

  for (const scanDot& dot : dots) {
    if (dot.angle >= min_angle && dot.angle <= max_angle) {
      distanceSum += dot.dist;
      n++;
    }
  }

  if (n == 0) {
    return 0;
  }

  return (distanceSum / n);
}

/**
 angle -- in degrees, with 0 being straight forward
*/
void setSteeringAngle(int angle) {
  // Uses an angular servo. Therefore, `angle` sets the target angle of the servo.
  // -180 - 180 -> 1000 - 2000

  int mappedValue = (int) (1000 * (angle + 180.0) / 360 + 1000);
  steeringServo.writeMicroseconds(mappedValue);
}

/**
 throttle -- a value from -1 to 1, corresponding to full speed forward or full speed backward
 this is roughly mapped so that 0 is at the stop value
*/
void setThrottle(float throttle) {
  int mappedValue = (int) ((throttle + 1) * 500 + 1000);
  if (mappedValue < 0) {
    mappedValue = 0;
  } else if (mappedValue >= 2000) {
    mappedValue = 2000;
  }
  Serial.print("throttle:"); Serial.println(mappedValue);
  // Uses a continuous servo. Therefore, `angle` sets the speed of the servo.
  throttleServo.writeMicroseconds(mappedValue);
}

/****** MAIN NAVIGATION CODE ****
 For now, simple wall avoidance algorithm that moves
 forward in the direction whose wall is furthest away.

 Forward angle is 0. Other angles are measured counterclockwise from this.
 ********************************/
void navigate(std::vector<scanDot> points) {

  double leftAverageDistance = getAverageDistance(points, 36, 54);
  double rightAverageDistance = getAverageDistance(points, 308, 324);
  Serial.print("left"); Serial.println(leftAverageDistance);
  Serial.print("right"); Serial.println(rightAverageDistance);
  if (leftAverageDistance > rightAverageDistance) {
    setSteeringAngle(45);
  } else {
    setSteeringAngle(-45);
  }
}

static void readPoints(void * parameter){
  while (true) {
    int result = lidar.cacheUltraCapsuledScanData();
    Serial.println(result, HEX);
  }
}

void setup() {
  pinMode(Pins::LIDAR,OUTPUT);
  digitalWrite(Pins::LIDAR, HIGH);

  steeringServo.attach(Pins::STEERING);
  throttleServo.attach(Pins::THROTTLE);
  Serial.begin(115200);
  esp_task_wdt_init(36000, false); // Turn off watchdog so core 0 task doesn't cause reset
  lidar.stopDevice(); // Reset the device to be sure that the status is good
  delay(1);
  
  if(!lidar.start(express)){
    Serial.println("failed to start");
    return;
  }

  // Start the express scan of the lidar
  /*
    Create Task on Specific Core (
      pvTaskCode = readPoints,
      pcName = "LidarPolling",
      usStackDepth = 65536,
      pvParameters = NULL,
      uxPriority = 2,
      pvCreatedTask = NULL,
      xCoreID = 0
    )
  */
  xTaskCreatePinnedToCore(readPoints, "LidarPolling", 65536, NULL, 2, NULL, 0);

}

void loop()
{
  // number of data points in cache: lidar._cached_scan_node_hq_count
  // float angle = (((float)_cached_scan_node_hq_buf[index].angle_z_q14) * 90.0 / 16384.0);
  // float distance = _cached_scan_node_hq_buf[index].dist_mm_q2 /4.0f;
  // each cache load contains a full 360 scan. If you slow down the rotations too much it will not fit and data will be lost (too many points per 360 deg for cache size allowable on ESP32)
  
  // READ
  // lidar.DebugPrintMeasurePoints(lidar._cached_scan_node_hq_count);
  
  // CONTROL/ACTUATE
  std::vector<scanDot> points = lidar.getPoints(lidar._cached_scan_node_hq_count);
  navigate(points);
  setThrottle(1);
  
  delay(1);
}