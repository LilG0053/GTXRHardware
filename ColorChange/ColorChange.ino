// NeoPixel test program showing use of the WHITE channel for RGBW
// pixels only (won't look correct on regular RGB NeoPixel strips).

#include <Adafruit_NeoPixel.h>
#ifdef __AVR__
  #include <avr/power.h> // Required for 16 MHz Adafruit Trinket
#endif

// Which pin on the Arduino is connected to the NeoPixels?
// On a Trinket or Gemma we suggest changing this to 1:
#define LED_PIN     2

// How many NeoPixels are attached to the Arduino?
#define LED_COUNT  9

// NeoPixel brightness, 0 (min) to 255 (max)
#define BRIGHTNESS 255 // Set BRIGHTNESS to about 1/5 (max = 255)
// R - 0 G - 1 B - 1

#define TIMEDELAY 50 // 50ms for 20Hz data rate

//RGB enum for pattern
enum Colors {
  R,
  G,
  C,
  P,
};

// Declare our NeoPixel strip object:
Adafruit_NeoPixel strip(LED_COUNT, LED_PIN, NEO_GRB + NEO_KHZ800);

uint32_t colors[] = {
  strip.Color(255, 0, 0), // Red
  strip.Color(0, 255, 0), // Green
  strip.Color(0, 20, 255), // Cyan
  strip.Color(128, 0, 255), // Purple
};

uint32_t LEDS[9][5] = {
  {R, G, C, G, R},
  {R, G, C, C, R},
  {R, G, C, P, R},
  {R, C, G, P, R},
  {R, C, G, C, R},
  {R, C, P, G, R},
  {R, P, G, P, R},
  {R, P, G, C, R},
  {R, P, C, P, R},
};

bool firstRun = true;
uint32_t lastlooptime;

// Argument 1 = Number of pixels in NeoPixel strip
// Argument 2 = Arduino pin number (most are valid)
// Argument 3 = Pixel type flags, add together as needed:
//   NEO_KHZ800  800 KHz bitstream (most NeoPixel products w/WS2812 LEDs)
//   NEO_KHZ400  400 KHz (classic 'v1' (not v2) FLORA pixels, WS2811 drivers)
//   NEO_GRB     Pixels are wired for GRB bitstream (most NeoPixel products)
//   NEO_RGB     Pixels are wired for RGB bitstream (v1 FLORA pixels, not v2)
//   NEO_RGBW    Pixels are wired for RGBW bitstream (NeoPixel RGBW products)

void setup() {
  // These lines are specifically to support the Adafruit Trinket 5V 16 MHz.
  // Any other board, you can remove this part (but no harm leaving it):
#if defined(__AVR_ATtiny85__) && (F_CPU == 16000000)
  clock_prescale_set(clock_div_1);
#endif
  // END of Trinket-specific code.

  strip.begin();           // INITIALIZE NeoPixel strip object (REQUIRED)
  strip.show();            // Turn OFF all pixels ASAP
  strip.setBrightness(BRIGHTNESS);
  lastlooptime = millis();
}

void loop() {
  // if (firstRun) {
    for (int i = 0; i < 5; i++) {
      for(int j = 0; j < 9; j++) {
        strip.setPixelColor(j, colors[LEDS[j][i]]);
        printf("Pixel %d : %d", j, colors[LEDS[j][i]]);
      }
      strip.show();
      while(millis() < lastlooptime + TIMEDELAY);
      lastlooptime = millis();
    }
    for (int i = 0; i < 9; i++) {
      strip.setPixelColor(i, colors[G]);
    }
    strip.show();
    while(millis() < lastlooptime + TIMEDELAY  +1000);
    lastlooptime = millis();
  //    firstRun = false;
  //  }
  
  // for (int k = 0; k < 9; k++) {
  //   strip.setPixelColor(k, colors[G]);
  // }
  // strip.show();
  // delay(500);
}
