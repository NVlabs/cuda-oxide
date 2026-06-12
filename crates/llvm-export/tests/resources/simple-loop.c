
int main() {

  int arr[5];
  int sum = 0;

  for (int i = 0; i < 5; i++) {
    arr[i] = i + 1;
  }
  for (int i = 0; i < 5; i++) {
    sum += arr[i];
  }

  return sum;
}
