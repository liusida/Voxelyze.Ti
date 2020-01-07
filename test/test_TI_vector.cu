#include "gtest/gtest.h"
#include <iostream>
#include <vector>
#include <time.h>

#include "TI_vector.h"

#ifdef _0

__global__ void test_time() {
  TI_vector<int> v;
  for (int i=0;i<100;i++)
    v.push_back(888);
}
TEST(time, gpu_time) {
  clock_t start, end;
  double cpu_time_used;
  start = clock();
  for (int i=0;i<100;i++) {
    test_time<<<1,100>>>();
    cudaDeviceSynchronize();
  }
  end = clock();
  cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
  printf("in gpu push. time used: %f.\n", cpu_time_used);
}
TEST(time, time) {
  printf("time");
  clock_t start, end;
  double cpu_time_used;
  
  std::vector<int> a;
  start = clock();
  for (int i=0;i<10000;i++) {
    a.push_back(1);
    TI_vector<int> b;
    b = a;
  }
  end = clock();
  cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
  printf("b = a; time used: %f.\n", cpu_time_used);

  a.clear();
  start = clock();
  for (int i=0;i<10000;i++) {
    a.push_back(1);
    TI_vector<int> b(a);
  }
  end = clock();
  cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
  printf("b(a); time used: %f.\n", cpu_time_used);

}
// The fixture for testing class Project1. From google test primer.
class TIVectorTest : public ::testing::Test {
protected:
  // You can remove any or all of the following functions if its body
  // is empty.

  TIVectorTest() {
    // You can do set-up work for each test here.
  }

  virtual ~TIVectorTest() {
    // You can do clean-up work that doesn't throw exceptions here.
  }

  // If the constructor and destructor are not enough for setting up
  // and cleaning up each test, you can define the following methods:
  virtual void SetUp() {
    // Code here will be called immediately after the constructor (right
    // before each test).
  }

  virtual void TearDown() {
    // Code here will be called immediately after each test (right
    // before the destructor).
  }

  // Objects declared here can be used by all tests in the test case for
  // Project1.
  TI_vector<int> p;
};

// Test case must be called the class above
// Also note: use TEST_F instead of TEST to access the test fixture (from google
// test primer)
TEST_F(TIVectorTest, init) {
    std::vector<int> a;
    a.push_back(1);
    a.push_back(2);
    
    p = a;
    int b[1024];
    cudaMemcpy(b, p.main, p.size()*sizeof(int), cudaMemcpyDeviceToHost);
    EXPECT_EQ(a.size(), 2);
    EXPECT_EQ(p.size(), 2);
    EXPECT_EQ(b[0], 1);
    EXPECT_EQ(b[1], 2);
}
#endif