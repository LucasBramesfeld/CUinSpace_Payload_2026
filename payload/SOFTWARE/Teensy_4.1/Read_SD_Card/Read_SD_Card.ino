#include <SD.h>
void printDirectory(File dir, int depth) {
  while (true) {
    File entry = dir.openNextFile();
    if (!entry) {
      break; // no more files
    }
    // Indentation for folders
    for (int i = 0; i < depth; i++) {
      Serial.print(" ");
    }
    Serial.print(entry.name());
    if (entry.isDirectory()) {

      Serial.println("/");
      printDirectory(entry, depth + 1);
    } else {
      Serial.print(" (");
      Serial.print(entry.size());
      Serial.println(" bytes)");
    }
    entry.close();
  }
}
void setup() {
  Serial.begin(9600);
  while (!Serial && millis() < 3000) {}
  if (!SD.begin(BUILTIN_SDCARD)) {
    Serial.println("SD init failed");
    return;
  }
  Serial.println("Listing SD card contents:");
  File root = SD.open("/");
  printDirectory(root, 0);
  root.close();
}
void loop() {}