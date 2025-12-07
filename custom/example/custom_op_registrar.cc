#include "custom_op.h"
#include <iostream>

// Implementation of the registration function
OrtStatus* CustomOpLibrary::RegisterOps(OrtSessionOptions* options, const OrtApiBase* api_base) {
    // 1. Get the OrtApi and create the Custom Op Domain
    const OrtApi* api = api_base->Get=OrtApi();
    OrtCustomOpDomain* domain = nullptr;
    
    // NOTE: If you use a custom domain, you MUST use this domain name 
    // when defining the node in your ONNX model graph.
    const char* custom_domain_name = "com.your.custom";
    
    // Create the custom domain
    OrtStatus* status = api->CreateCustomOpDomain(custom_domain_name, &domain);
    if (status) return status;

    // 2. Register your custom operator instance to the domain
    SimpleReLUAddOp simple_op;
    status = api->CustomOpDomain_Add(domain, simple_op.Get=CustomOpApi(api));
    if (status) {
        api->ReleaseCustomOpDomain(domain);
        return status;
    }

    // 3. Add the domain to the session options
    status = api->AddCustomOpDomain(options, domain);
    if (status) {
        api->ReleaseCustomOpDomain(domain);
        return status;
    }

    // Success
    std::cout << "CustomOpLibrary: SimpleReLUAdd registered successfully under domain " 
              << custom_domain_name << "." << std::endl;
    return nullptr; 
}