#include "blerp.h"

/*
 *    ^    ^                    ^
 *    |    |                    |
 *    |    |                    |
 * y2 |--- p12 ---------------- p22 ---->
 *    |    |                    |
 *    |    |                    |
 *    |    |                    |
 * y  |    |    . blerp         |
 *    |    |                    |
 *    |    |                    |
 *    |    |                    |
 *    |    |                    |
 *    |    |                    |
 * y1 |--- p11 ---------------- p21 ---->
 *    |    |                    |
 *    |    |                    |
 *    __________________________________>
 *  0      x1   x               x2
 */

double blerp(
    double x, double y,
    double x1, double x2,
    double y1, double y2,
    double p11, double p12,
    double p21, double p22
) {
  return (
      + (p11 * (x2 - x ) + p21 * (x  - x1)) * (y2 - y )
      + (p12 * (x2 - x ) + p22 * (x  - x1)) * (y  - y1)
  ) / ((x2 - x1) * (y2 - y1));
}

double gradBlerpX(
    double x, double y,
    double x1, double x2,
    double y1, double y2,
    double p11, double p12,
    double p21, double p22
) {
  return (
      - p11 * (y2 - y )
      + p21 * (y2 - y )
      - p12 * (y  - y1)
      + p22 * (y  - y1)
  ) / ((x2 - x1) * (y2 - y1));
}

double gradBlerpY(
    double x, double y,
    double x1, double x2,
    double y1, double y2,
    double p11, double p12,
    double p21, double p22
) {
  return (
      - p11 * (x2 - x )
      - p21 * (x  - x1)
      + p12 * (x2 - x )
      + p22 * (x  - x1)
  ) / ((x2 - x1) * (y2 - y1));
}

double gradBlerpP11(
    double x, double y,
    double x1, double x2,
    double y1, double y2,
    double p11, double p12,
    double p21, double p22
) {
  return (
      (x2 - x ) * (y2 - y )
  ) / ((x2 - x1) * (y2 - y1));
}

double gradBlerpP12(
    double x, double y,
    double x1, double x2,
    double y1, double y2,
    double p11, double p12,
    double p21, double p22
) {
  return (
      (x2 - x ) * (y  - y1)
  ) / ((x2 - x1) * (y2 - y1));
}

double gradBlerpP21(
    double x, double y,
    double x1, double x2,
    double y1, double y2,
    double p11, double p12,
    double p21, double p22
) {
  return (
      (x  - x1) * (y2 - y )
  ) / ((x2 - x1) * (y2 - y1));
}

double gradBlerpP22(
    double x, double y,
    double x1, double x2,
    double y1, double y2,
    double p11, double p12,
    double p21, double p22
) {
  return (
      (x  - x1) * (y  - y1)
  ) / ((x2 - x1) * (y2 - y1));
}
