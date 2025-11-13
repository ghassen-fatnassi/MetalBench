// papi_test.cpp
#include <iostream>
#include <papi.h>

int main() {
    int ret = PAPI_library_init(PAPI_VER_CURRENT);
    if (ret != PAPI_OK) {
        std::cout << "PAPI init failed: " << PAPI_strerror(ret) << std::endl;
        return 1;
    }
    std::cout << "PAPI initialized successfully!" << std::endl;
    
    // Test creating an event set
    int EventSet = PAPI_NULL;
    if (PAPI_create_eventset(&EventSet) == PAPI_OK) {
        std::cout << "EventSet created successfully!" << std::endl;
        PAPI_cleanup_eventset(EventSet);
    }
    
    PAPI_shutdown();
    return 0;
}