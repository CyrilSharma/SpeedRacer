/*
 *  @author KKest
 *		@created 10.01.2022
 *
 * Library to control an rpLidar S2
 *
 */
#include "navigator.h"
#include "Arduino.h"
#include <bits/stdc++.h>

int navigator::getAngle(int16_t count, int16_t min, int16_t max)
{
  using std::vector;

  vector<int> values;
  for (int pos = 0; pos < (int)count; ++pos) {
      scanDot dot;
      if (!_cached_scan_node_hq_buf[pos].dist_mm_q2) continue;
      //dot.quality = _cached_scan_node_hq_buf[pos].quality; //quality is broken for some reason
      dot.angle = (((float)_cached_scan_node_hq_buf[pos].angle_z_q14) * 90.0 / 16384.0);
      dot.dist = _cached_scan_node_hq_buf[pos].dist_mm_q2 /4.0f;
      if (dot.angle >=min && dot.angle <=max){
          d
          return dot.dist;
      }
  }
  return 0;
}
