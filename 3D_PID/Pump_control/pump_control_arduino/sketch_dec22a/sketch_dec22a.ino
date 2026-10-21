/*************************************************
 *  Arduino DC Motor Control Example
 *  - Speed input: 0–100 (%)
 *  - Internal mapping to PWM 0–255
 *  - With minimum start PWM compensation
 *************************************************/

// ===== 引脚定义 =====
const uint8_t E1 = 10;   // PWM 引脚
const uint8_t E2 = 11;   // PWM 引脚
const uint8_t M1 = 12;   // 方向控制
const uint8_t M2 = 13;   // 方向控制

// ===== PWM 参数 =====
const uint8_t PWM_MIN = 80;   // 最小可起转 PWM（需按你的泵微调）
const uint8_t PWM_MAX = 255;

// ===== 百分比 → PWM 映射 =====
uint8_t percentToPWM(uint8_t percent)
{
  if (percent == 0) return 0;
  if (percent >= 100) return PWM_MAX;

  uint16_t pwm = (uint16_t)percent * PWM_MAX / 100;

  // 起转死区补偿
  if (pwm < PWM_MIN) pwm = PWM_MIN;

  return (uint8_t)pwm;
}

// ===== 电机控制接口（核心）=====
void setMotor(bool direction, uint8_t speedPercent)
{
  uint8_t pwm = percentToPWM(speedPercent);

  // 设置方向
  digitalWrite(M1, direction ? HIGH : LOW);
  digitalWrite(M2, direction ? HIGH : LOW);

  // 设置速度
  analogWrite(E1, pwm);
  analogWrite(E2, pwm);
}

// ===== 停止电机 =====
void stopMotor()
{
  analogWrite(E1, 0);
  analogWrite(E2, 0);
}

// ===== 初始化 =====
void setup()
{
  pinMode(E1, OUTPUT);
  pinMode(E2, OUTPUT);
  pinMode(M1, OUTPUT);
  pinMode(M2, OUTPUT);

  stopMotor();  // 上电默认安全停止
}

// ===== 示例主循环 =====
void loop()
{
  // 正转 35%
  setMotor(true, 35);
  delay(1000);

  // 正转 80%
  setMotor(true, 80);
  delay(1000);

  // 反转 35%
  setMotor(false, 35);
  delay(1000);

  // 反转 80%
  setMotor(false, 80);
  delay(1000);

  // 停止
  stopMotor();
  delay(1000);
}
