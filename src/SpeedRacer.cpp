#include <Arduino.h>
#include "rpLidar.h"
#include "rpLidarTypes.h"
#include <esp_task_wdt.h>

rpLidar lidar(&Serial2,115200,13,12);


double getAverageDistance(int16_t count, float min_angle, float max_angle)
{
  float distanceSum = 0;
  int distanceCount = 0;

  for (int pos = 0; pos < (int)count; ++pos) {
      scanDot dot;
      if (!_cached_scan_node_hq_buf[pos].dist_mm_q2) continue;
      //dot.quality = _cached_scan_node_hq_buf[pos].quality; //quality is broken for some reason
      dot.angle = (((float)_cached_scan_node_hq_buf[pos].angle_z_q14) * 90.0 / 16384.0);
      dot.dist = _cached_scan_node_hq_buf[pos].dist_mm_q2 /4.0f;
      
      // Might be some trippy stuff with angles...
      if (dot.angle >= min_angle && dot.angle <= max_angle) {
        distanceSum += dot.dist;
        distanceCount++;
      }
  }

  if (distanceCount == 0) {
    return 0;
  }

  return (distanceSum / distanceCount);
}
static void readPoints(void * parameter){
  while(true){
    int result = lidar.cacheUltraCapsuledScanData();
    Serial.println(result,HEX);
  }
}
void setup() {

  pinMode(19,OUTPUT);
  digitalWrite(19,HIGH);
  Serial.begin(115200);
  esp_task_wdt_init(36000, false); //turn off watchdog so core 0 task doesn't cause reset
  lidar.stopDevice(); //reset the device to be sure that the status is good
  delay(1);
  if(!lidar.start(express)){
    Serial.println("failed to start");
    return;
  } //start the express scan of the lidar\  esp_task_wdt_init(36000, false); //turn off watchdog so core 0 task doesn't cause reset

  xTaskCreatePinnedToCore(readPoints, "LidarPolling", 65536, NULL, 2, NULL, 0);

}

void loop()
{
// number of data points in cache: lidar._cached_scan_node_hq_count
// float angle = (((float)_cached_scan_node_hq_buf[index].angle_z_q14) * 90.0 / 16384.0);
// float distance = _cached_scan_node_hq_buf[index].dist_mm_q2 /4.0f;
// each cache load contains a full 360 scan. If you slow down the rotations too much it will not fit and data will be lost (too many points per 360 deg for cache size allowable on ESP32)
 lidar.DebugPrintMeasurePoints(lidar._cached_scan_node_hq_count);
 delay(1);
}