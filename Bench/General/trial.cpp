#include "build/onnxruntime-linux-x64-1.23.2/include/onnxruntime_cxx_api.h"
#include <chrono>
#include <iostream>
#include <sys/sysinfo.h>
#include <thread>
#include <random>

using namespace std;
using clk=std::chrono::high_resolution_clock;

void get_system_info()
{
  cout<< "=== SYSTEM INFO ===\n";
  cout<< " vCPUs/how many execution contexts there are: " << std::thread::hardware_concurrency()<<"\n";
  // need to get the useful information about the cpu
  struct sysinfo info;
  cout<< "Total Usable Main Memory: " << (info.totalram/(1<<20))<<"\n";
  cout<< "Total SWAP: " << (info.totalswap/(1<<20))<<"\n";
  cout<< "Total High Memory: " << (info.totalhigh/(1<<20))<<"\n";
  cout<< "===================\n";
}
void get_onnx_benchmark()
{
  Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "benchamrk");
  Ort::SessionOptions options;

  //understand the next section in depth
  //section start
  options.SetIntraOpNumThreads(4);
  options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_BASIC); // I GOTTA TRY ALL THE LEVELS 
  options.EnableCpuMemArena();
  options.EnableMemPattern();
  options.EnableProfiling("onnxruntime_profile.json");
  //section end

  string model_path = "../Models/yolo12n.onnx";
  Ort::Session session(env, model_path.c_str(), options);
  // we put index 0 in 
  auto input_shape =session.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
  for (auto &d : input_shape)
  {
    if (d<=0) {d=1;}
     // this is a check on the batch size dimension, 
     // which is dynamic as onnx support dynamic batch size, 
     // so it sets it to -1 on initialization so the runtime accepts anytype of batch_size
     // the same principle applies to any dimension of input that's dynamic and unkown so that's why we loop
  }

  size_t tensor_size =1;
  for (auto d : input_shape) 
  {
    tensor_size *=d; 
    // it's not multiplied by 4 , since size here doesn't really mean the numebr of bytes but the number of elements in the tensor
  };

  // generation of random input START
  vector<float> input_data(tensor_size);
  mt19937 gen(43);
  uniform_real_distribution<float> dist(0.f,1.f);
  for(auto &x:input_data) {x=dist(gen);}
  // generation of random input END

  Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator,OrtMemTypeDefault);
  Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
                                                          mem_info, 
                                                          input_data.data(),
                                                          tensor_size, 
                                                          input_shape.data(), 
                                                          input_shape.size());
  auto input_names = session.GetInputNames();
  auto output_names = session.GetOutputNames();
  cout<<"input_names size :: "<<input_names.size()<<"\n";
  cout<<"output_names size :: "<<output_names.size()<<"\n";
  vector<const char*> in_c, out_c;
  for(auto &s : input_names){in_c.push_back(s.c_str());}
  for(auto &s : output_names){out_c.push_back(s.c_str());}

  //this is how to do a run in onnx , one forward pass over the model
  session.Run(
              Ort::RunOptions{nullptr},
              in_c.data(),&input_tensor, input_names.size(), out_c.data(), out_c.size());
}

int main()
{
  cout << "benchmarking general";
  get_system_info();
  get_onnx_benchmark();
}