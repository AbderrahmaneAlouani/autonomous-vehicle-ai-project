// Autonomous Vehicle Main Controller

// Motor control pins
const int motorLeft = 5;
const int motorRight = 6;
const int steeringServo = 9;

// Sensor pins
const int trigPin = 10;
const int echoPin = 11;
const int irLeft = A0;
const int irRight = A1;

void setup()
{
    Serial.begin(9600);

    // Initialize pins
    pinMode(motorLeft, OUTPUT);
    pinMode(motorRight, OUTPUT);
    pinMode(steeringServo, OUTPUT);
    pinMode(trigPin, OUTPUT);
    pinMode(echoPin, INPUT);

    Serial.println("Autonomous Vehicle Ready");
}

void loop()
{
    // Read sensors
    long distance = readUltrasonic();
    int leftIR = analogRead(irLeft);
    int rightIR = analogRead(irRight);

    // Send sensor data
    Serial.print("DIST:");
    Serial.print(distance);
    Serial.print(",LIR:");
    Serial.print(leftIR);
    Serial.print(",RIR:");
    Serial.println(rightIR);

    delay(100);
}

long readUltrasonic()
{
    digitalWrite(trigPin, LOW);
    delayMicroseconds(2);
    digitalWrite(trigPin, HIGH);
    delayMicroseconds(10);
    digitalWrite(trigPin, LOW);

    long duration = pulseIn(echoPin, HIGH);
    return duration * 0.034 / 2;
}