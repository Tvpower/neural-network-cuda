//
// Created by tvpower on 5/11/25.
//
#include "gtest/gtest.h" // testing framework
#include "utils/utils.h"  // headers for testing

TEST(UtilsTest, ExampleTest) {
    ASSERT_EQ(1, 1);  // A simple test
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}