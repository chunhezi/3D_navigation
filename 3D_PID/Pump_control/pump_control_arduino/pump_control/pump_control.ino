/*************************************************
 * Arduino Pump Control via Serial (Python)
 * Commands (newline terminated):
 *   SET <dir> <speed>
 *     - dir: 1 forward, 0 reverse
 *     - speed: 0..100 (%)
 *   STOP
 *   PING
 * Replies:
 *   OK / ERR / PONG
 *************************************************/

const uint8_t E1 = 10;
const uint8_t E2 = 11;
const uint8_t M1 = 12;
const uint8_t M2 = 13;

const uint8_t PWM_MIN = 100;
const uint8_t PWM_MAX = 255;

uint8_t percentToPWM(uint8_t percent)
{
  if (percent == 0) return 0;
  if (percent >= 100) return PWM_MAX;

  uint16_t pwm = (uint16_t)percent * PWM_MAX / 100;
  if (pwm < PWM_MIN) pwm = PWM_MIN;
  return (uint8_t)pwm;
}

void setMotor(bool direction, uint8_t speedPercent)
{
  uint8_t pwm = percentToPWM(speedPercent);
  digitalWrite(M1, direction ? HIGH : LOW);
  digitalWrite(M2, direction ? HIGH : LOW);
  analogWrite(E1, pwm);
  analogWrite(E2, pwm);
}

void stopMotor()
{
  analogWrite(E1, 0);
  analogWrite(E2, 0);
}

void setup()
{
  pinMode(E1, OUTPUT);
  pinMode(E2, OUTPUT);
  pinMode(M1, OUTPUT);
  pinMode(M2, OUTPUT);

  stopMotor();
  Serial.begin(115200);
  Serial.println("READY");
}

void loop()
{
  if (!Serial.available()) return;

  String line = Serial.readStringUntil('\n');
  line.trim();
  if (line.length() == 0) return;

  if (line == "PING") {
    Serial.println("PONG");
    return;
  }

  if (line == "STOP") {
    stopMotor();
    Serial.println("OK");
    return;
  }

  // SET <dir> <speed>
  if (line.startsWith("SET")) {
    // 允许 "SET 1 35" / "SET 0 80"
    int firstSpace = line.indexOf(' ');
    int secondSpace = line.indexOf(' ', firstSpace + 1);
    if (firstSpace < 0 || secondSpace < 0) {
      Serial.println("ERR");
      return;
    }

    int dir = line.substring(firstSpace + 1, secondSpace).toInt();
    int spd = line.substring(secondSpace + 1).toInt();
    if (spd < 0) spd = 0;
    if (spd > 100) spd = 100;

    setMotor(dir != 0, (uint8_t)spd);
    Serial.println("OK");
    return;
  }

  Serial.println("ERR");
}
