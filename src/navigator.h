/*
 *  @author KKest
 *		@created 10.01.2022
 *	
 * Library to control an rpLidar S2
 *
 */
 
#ifndef navigator_h
#define navigator_h

#include "rpLidarTypes.h"
#include <vector>

class Navigator{
	public:
	
	/**
	 * Construcor of Class
    */
	Navigator(HardwareSerial *_serial,uint32_t baud, int rx, int tx);
	
    /* example func bc i forgot how c++ works */
	bool start(uint8_t _mode);

    vector<int> getAngle(int16_t count, int16_t min, int16_t max);
	
	/* example thing */
	HardwareSerial *serial;		///< pointer to HardwareSerial USART 
};

#endif
