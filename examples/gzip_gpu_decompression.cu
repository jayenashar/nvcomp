/*
 * Copyright (c) 2020-2021, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
 #include "BatchData.h"
 #include "zlib.h"
 #include "nvcomp/gzip.h"

// modelled after BatchData(BatchDataCPU, bool) and operator==(BatchDataCPU, BatchData)
// seems i didn't need to write this and could have just used BatchDataCPU(void**, size_t*, uint8_t*, size_t, bool)
BatchDataCPU::BatchDataCPU(const BatchData& batch_data, bool copy_data) :
  m_ptrs(),
  m_sizes(),
  m_data(),
  m_size()
{
  m_size = batch_data.size();
  m_sizes = std::vector<size_t>(batch_data.size());
  CUDA_CHECK(cudaMemcpy(
      m_sizes.data(),
      batch_data.sizes(),
      batch_data.size() * sizeof(size_t),
      cudaMemcpyDeviceToHost));

  size_t data_size = std::accumulate(
        sizes(),
        sizes() + size(),
        static_cast<size_t>(0));
  m_data = std::vector<uint8_t>(data_size);

  size_t offset = 0;
  std::vector<void*> ptrs(size());
  for (size_t i = 0; i < size(); ++i) {
    ptrs[i] = data() + offset;
    offset += sizes()[i];
  }
  m_ptrs = std::vector<void*>(ptrs);

  if (copy_data) {
    std::vector<void*> src(batch_data.size());
    CUDA_CHECK(cudaMemcpy(
        src.data(),
        batch_data.ptrs(),
        batch_data.size() * sizeof(void*),
        cudaMemcpyDeviceToHost));

    const size_t* bytes = sizes();
    for (size_t i = 0; i < size(); ++i) {
      CUDA_CHECK(
          cudaMemcpy(ptrs[i], src[i], bytes[i], cudaMemcpyDeviceToHost));
    }
  }
}

 // Benchmark performance from the binary data file fname
 static void run_example(const std::vector<std::vector<char>>& data)
 {
   size_t total_bytes = 0;
   std::vector<size_t> comp_sizes;
   std::vector<size_t> decomp_sizes;
   size_t max_decomp_size = 0;
   for (const std::vector<char>& part : data) {
     comp_sizes.push_back(part.size());
     // get uncompressed size of file from gzip footer
     size_t size = *(size_t*)(part.data() + part.size() - 4);
     total_bytes += size;
     decomp_sizes.push_back(size);
     max_decomp_size = std::max(max_decomp_size, size);
   }
 
   std::cout << "----------" << std::endl;
   std::cout << "files: " << data.size() << std::endl;
   std::cout << "uncompressed (B): " << total_bytes << std::endl;
 
   // seems to be the maximum chunk size
   const size_t chunk_size = 1 << 16;
 
   // build up input batch on CPU
   BatchDataCPU compress_data_cpu(data, comp_sizes);
   std::cout << "chunks: " << compress_data_cpu.size() << std::endl;
 
   // compute compression ratio
   size_t* compressed_sizes_host = compress_data_cpu.sizes();
   size_t comp_bytes = 0;
   for (size_t i = 0; i < compress_data_cpu.size(); ++i)
     comp_bytes += compressed_sizes_host[i];
 
   std::cout << "comp_size: " << comp_bytes
             << ", compressed ratio: " << std::fixed << std::setprecision(2)
             << (double)total_bytes / comp_bytes << std::endl;
 
   // Copy compressed data to GPU
   BatchData compress_data(compress_data_cpu, true);
 
   // Allocate and build up decompression batch on GPU
   BatchData decomp_data(total_bytes, decomp_sizes);
 
   // Create CUDA stream
   cudaStream_t stream;
   cudaStreamCreate(&stream);
 
   // CUDA events to measure decompression time
   cudaEvent_t start, end;
   cudaEventCreate(&start);
   cudaEventCreate(&end);

   // deflate GPU decompression
   size_t decomp_temp_bytes;
   nvcompStatus_t status = nvcompBatchedGzipDecompressGetTempSize(
       compress_data.size(), max_decomp_size, &decomp_temp_bytes);
   if (status != nvcompSuccess) {
     throw std::runtime_error("nvcompBatchedGzipDecompressGetTempSize() failed.");
   }
 
   void* d_decomp_temp;
   CUDA_CHECK(cudaMalloc(&d_decomp_temp, decomp_temp_bytes));
 
   size_t* d_decomp_sizes;
   CUDA_CHECK(cudaMalloc(&d_decomp_sizes, decomp_data.size() * sizeof(size_t)));
 
   nvcompStatus_t* d_status_ptrs;
   CUDA_CHECK(cudaMalloc(&d_status_ptrs, decomp_data.size() * sizeof(nvcompStatus_t)));
 
   CUDA_CHECK(cudaStreamSynchronize(stream));
 
   // Run decompression
   status = nvcompBatchedGzipDecompressAsync(
       compress_data.ptrs(),
       compress_data.sizes(),
       decomp_data.sizes(),
       d_decomp_sizes,
       compress_data.size(),
       d_decomp_temp,
       decomp_temp_bytes,
       decomp_data.ptrs(),
       d_status_ptrs,
       stream);
   if( status != nvcompSuccess){
     throw std::runtime_error("ERROR: nvcompBatchedGzipDecompressAsync() not successful");
   }

   BatchDataCPU decomp_data_cpu(decomp_data, true);
   std::cout << "chunks: " << decomp_data_cpu.size() << std::endl;

   // Re-run decompression to get throughput
   cudaEventRecord(start, stream);
   status = nvcompBatchedGzipDecompressAsync(
     compress_data.ptrs(),
     compress_data.sizes(),
     decomp_data.sizes(),
     d_decomp_sizes,
     compress_data.size(),
     d_decomp_temp,
     decomp_temp_bytes,
     decomp_data.ptrs(),
     d_status_ptrs,
     stream);
   cudaEventRecord(end, stream);
   if( status != nvcompSuccess){
     throw std::runtime_error("ERROR: nvcompBatchedGzipDecompressAsync() not successful");
   }
 
   CUDA_CHECK(cudaStreamSynchronize(stream));
 
   float ms;
   cudaEventElapsedTime(&ms, start, end);
 
   double decompression_throughput = ((double)total_bytes / ms) * 1e-6;
   std::cout << "decompression throughput (GB/s): " << decompression_throughput
             << std::endl;
   std::cout << "ms: " << ms << std::endl;
 
   cudaFree(d_decomp_temp);
 
   cudaEventDestroy(start);
   cudaEventDestroy(end);
   cudaStreamDestroy(stream);
 }
 
 std::vector<char> readFile(const std::string& filename)
 {
   std::vector<char> buffer(4096);
   std::vector<char> host_data;
 
   std::ifstream fin(filename, std::ifstream::binary);
   fin.exceptions(std::ifstream::failbit | std::ifstream::badbit);
 
   size_t num;
   do {
     num = fin.readsome(buffer.data(), buffer.size());
     host_data.insert(host_data.end(), buffer.begin(), buffer.begin() + num);
   } while (num > 0);
 
   return host_data;
 }
 
 std::vector<std::vector<char>>
 multi_file(const std::vector<std::string>& filenames)
 {
   std::vector<std::vector<char>> split_data;
 
   for (auto const& filename : filenames) {
     split_data.emplace_back(readFile(filename));
   }
 
   return split_data;
 }
 
 int main(int argc, char* argv[])
 {
    std::vector<std::string> file_names(argc - 1);

    if (argc == 1) {
      std::cerr << "Must specify at least one file." << std::endl;
      return 1;
    }
  
    // if `-f` is specified, assume single file mode
    if (strcmp(argv[1], "-f") == 0) {
      if (argc == 2) {
        std::cerr << "Missing file name following '-f'" << std::endl;
        return 1;
      } else if (argc > 3) {
        std::cerr << "Unknown extra arguments with '-f'." << std::endl;
        return 1;
      }
  
      file_names = {argv[2]};
    } else {
      // multi-file mode
      for (int i = 1; i < argc; ++i) {
        file_names[i - 1] = argv[i];
      }
    }
  
    auto data = multi_file(file_names);
  
    run_example(data);
 
   return 0;
 }
 