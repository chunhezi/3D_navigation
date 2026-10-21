/*************************************************
 * Constant Water Flow Output
 * 功能：
 *   - 上电后直接输出恒定水流
 *   - 扰动方向和扰动强度在程序中固定设置
 *   - 不使用串口通信
 *
 * 可调参数：
 *   FLOW_DIR     : 1 正向, 0 反向
 *   FLOW_PERCENT : 0~100，对应扰动强度
 *************************************************/

const uint8_t E1 = 10;
const uint8_t E2 = 11;
const uint8_t M1 = 12;
const uint8_t M2 = 13;

const uint8_t PWM_MIN = 100;   // 防止占空比过小带不动泵
const uint8_t PWM_MAX = 255;

// ====== 这里直接设置恒值水流参数 ======
const bool FLOW_DIR = 1;        // 1: 正向, 0: 反向
const uint8_t FLOW_PERCENT = 20; // 扰动强度：0~100
// ====================================

uint8_t percentToPWM(uint8_t percent)
{
  if (percent == 0) return 0;
  if (percent >= 100) return PWM_MAX;

  uint16_t pwm = (uint16_t)percent * PWM_MAX / 100;

  // 非零时保证至少达到最小有效PWM
  if (pwm < PWM_MIN) pwm = PWM_MIN;

  return (uint8_t)pwm;
}

void setFlow(bool direction, uint8_t speedPercent)
{
  uint8_t pwm = percentToPWM(speedPercent);

  // 设置方向
  digitalWrite(M1, direction ? HIGH : LOW);
  digitalWrite(M2, direction ? HIGH : LOW);

  // 设置强度（PWM）
  analogWrite(E1, pwm);
  analogWrite(E2, pwm);
}

void stopFlow()
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

  // 上电先停一下，避免误动作
  stopFlow();
  delay(500);

  // 输出设定的恒值水流
  setFlow(FLOW_DIR, FLOW_PERCENT);
}

void loop()
{
  // 恒值输出，不需要循环控制
}