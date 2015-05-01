// Copyright 2014 BVLC and contributors.

#include <stdint.h>
#include <fcntl.h>
#include <google/protobuf/text_format.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/io/coded_stream.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc.hpp>

#include <algorithm>
#include <string>
#include <vector>
#include <fstream>  // NOLINT(readability/streams)
#include <iostream>

#include "caffe/common.hpp"
#include "caffe/util/io.hpp"
#include "caffe/proto/caffe.pb.h"

using std::fstream;
using std::ios;
using std::max;
using std::string;
using google::protobuf::io::FileInputStream;
using google::protobuf::io::FileOutputStream;
using google::protobuf::io::ZeroCopyInputStream;
using google::protobuf::io::CodedInputStream;
using google::protobuf::io::ZeroCopyOutputStream;
using google::protobuf::io::CodedOutputStream;
using google::protobuf::Message;

namespace caffe {

bool ReadProtoFromTextFile(const char* filename, Message* proto) {
  int fd = open(filename, O_RDONLY);
  CHECK_NE(fd, -1) << "File not found: " << filename;
  FileInputStream* input = new FileInputStream(fd);
  bool success = google::protobuf::TextFormat::Parse(input, proto);
  delete input;
  close(fd);
  return success;
}

void WriteProtoToTextFile(const Message& proto, const char* filename) {
  int fd = open(filename, O_WRONLY | O_CREAT | O_TRUNC, 0644);
  FileOutputStream* output = new FileOutputStream(fd);
  CHECK(google::protobuf::TextFormat::Print(proto, output));
  delete output;
  close(fd);
}

bool ReadProtoFromBinaryFile(const char* filename, Message* proto) {
  int fd = open(filename, O_RDONLY);
  CHECK_NE(fd, -1) << "File not found: " << filename;
  ZeroCopyInputStream* raw_input = new FileInputStream(fd);
  CodedInputStream* coded_input = new CodedInputStream(raw_input);
  coded_input->SetTotalBytesLimit(1073741824, 536870912);

  bool success = proto->ParseFromCodedStream(coded_input);

  delete coded_input;
  delete raw_input;
  close(fd);
  return success;
}

void WriteProtoToBinaryFile(const Message& proto, const char* filename) {
  fstream output(filename, ios::out | ios::trunc | ios::binary);
  CHECK(proto.SerializeToOstream(&output));
}

bool ReadImageToDatum(const string& filename, const int label,
    const int height, const int width, Datum* datum) {
  cv::Mat cv_img;
  if (height > 0 && width > 0) {
    cv::Mat cv_img_origin = cv::imread(filename, CV_LOAD_IMAGE_COLOR);
    cv::resize(cv_img_origin, cv_img, cv::Size(height, width));
  } else {
    cv_img = cv::imread(filename, CV_LOAD_IMAGE_COLOR);
  }
  if (!cv_img.data) {
    LOG(ERROR) << "Could not open or find file " << filename;
    return false;
  }
  datum->set_channels(3);
  datum->set_height(cv_img.rows);
  datum->set_width(cv_img.cols);
  datum->set_label(label);
  datum->clear_data();
  datum->clear_float_data();
  string* datum_string = datum->mutable_data();
  for (int c = 0; c < 3; ++c) {
    for (int h = 0; h < cv_img.rows; ++h) {
      for (int w = 0; w < cv_img.cols; ++w) {
        datum_string->push_back(
            static_cast<char>(cv_img.at<cv::Vec3b>(h, w)[c]));
      }
    }
  }
  return true;
}

bool ReadAndResizeImageToDatum(const string& filename, const int label,
    const int height, const int width, int new_height, int new_width,
    Datum* datum) {
  cv::Mat cv_img;
  if (height > 0 && width > 0) {
    cv::Mat cv_img_origin = cv::imread(filename, CV_LOAD_IMAGE_COLOR);
    cv::resize(cv_img_origin, cv_img, cv::Size(height, width));
  } else {
    cv_img = cv::imread(filename, CV_LOAD_IMAGE_COLOR);
  }
  LOG(INFO) << "Opened image" << std::endl;
  if (!cv_img.data) {
    LOG(ERROR) << "Could not open or find file " << filename;
    return false;
  }

  int old_height = cv_img.rows;
  int old_width = cv_img.cols;

  cv::Mat normalized;

  new_height = std::max(new_height, new_width);
  new_width = new_height;

  if (new_height > old_height || new_width > old_width) {
  	normalized.create(new_height, new_width, cv_img.type());
    std::cout << "Padding " << cv_img.size() << " to " << normalized.size() << std::endl;
  	normalized.setTo(cv::Scalar::all(0));
  	cv::Rect roi( cv::Point( 0, 0 ), cv_img.size() );
  	cv_img.copyTo( normalized( roi ) );
  } else {
  	cv::Mat padded;
  	int max_dim = std::max(cv_img.rows, cv_img.cols);
  	padded.create(max_dim, max_dim, cv_img.type());
  	padded.setTo(cv::Scalar::all(0));
    std::cout << "Downscaling " << cv_img.size() << " to " << padded.size() << std::endl;
  	cv::Rect roi( cv::Point( 0, 0 ), cv_img.size() );
  	cv_img.copyTo( padded( roi ) );

  	cv::Mat resized(cv::Size(new_height, new_width), cv_img.type());
    std::cout << "Resizing " << padded.size() << " to " << resized.size() << std::endl;
  	cv::resize(padded, resized, resized.size(), 0, 0, CV_INTER_AREA);
  	normalized = resized;
  }

//  std::cout << "Displaying image" << std::endl << std::cout.flush();
//  cv::namedWindow( "Display window" );
//  cv::imshow( "Display window", normalized );
//  cv::waitKey(0);

  datum->set_channels(3);
  datum->set_height(normalized.rows);
  datum->set_width(normalized.cols);
  datum->set_label(label);
  datum->clear_data();
  datum->clear_float_data();
  string* datum_string = datum->mutable_data();
  for (int c = 0; c < 3; ++c) {
    for (int h = 0; h < normalized.rows; ++h) {
      for (int w = 0; w < normalized.cols; ++w) {
        datum_string->push_back(
            static_cast<char>(normalized.at<cv::Vec3b>(h, w)[c]));
      }
    }
  }

  cv_img.release();
  normalized.release();
  return true;
}

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
cv::Mat pad_with_black(cv::Mat& img, int new_rows, int new_cols) {

	cv::Mat padded;
	std::cout << "img.rows " << img.rows << " img.cols " << img.cols << std::endl;

	std::cout << "Creating padded image " << new_rows << " " << new_cols << std::endl << std::cout.flush();
	padded.create(new_rows, new_cols, img.type());

	std::cout << "Setting to black" << std::endl << std::cout.flush();
  padded.setTo(cv::Scalar::all(0));

	std::cout << "Copying original image to padded image" << std::endl << std::cout.flush();
	//img.copyTo(padded(cv::Rect(0, 0, img.rows-1, img.cols-1)));

	cv::Rect roi( cv::Point( 0, 0 ), img.size() );
	img.copyTo( padded( roi ) );

//	std::cout << "Creating rect" << std::endl << std::cout.flush();
//	cv::Rect rect = cv::Rect(0, 0, img.rows, img.cols);
//	std::cout << "Creating roi" << std::endl << std::cout.flush();
//	cv::Mat roi = padded(rect);
//	std::cout << "Copying original image to padded image" << std::endl << std::cout.flush();
//	img.copyTo(roi);

  cv::namedWindow( "Display window", cv::WINDOW_NORMAL );// Create a window for display.
  cv::imshow( "Display window", padded );                   // Show our image inside it.
  cv::waitKey(0);

	std::cout << "Done" << std::endl << std::cout.flush();
	return padded;
}

// Verifies format of data stored in HDF5 file and reshapes blob accordingly.
template <typename Dtype>
void hdf5_load_nd_dataset_helper(
    hid_t file_id, const char* dataset_name_, int min_dim, int max_dim,
    Blob<Dtype>* blob) {
  // Verify that the number of dimensions is in the accepted range.
  herr_t status;
  int ndims;
  status = H5LTget_dataset_ndims(file_id, dataset_name_, &ndims);
  CHECK_GE(ndims, min_dim);
  CHECK_LE(ndims, max_dim);

  // Verify that the data format is what we expect: float or double.
  std::vector<hsize_t> dims(ndims);
  H5T_class_t class_;
  status = H5LTget_dataset_info(
      file_id, dataset_name_, dims.data(), &class_, NULL);
  CHECK_EQ(class_, H5T_FLOAT) << "Expected float or double data";

  blob->Reshape(
    dims[0],
    (dims.size() > 1) ? dims[1] : 1,
    (dims.size() > 2) ? dims[2] : 1,
    (dims.size() > 3) ? dims[3] : 1);
}

template <>
void hdf5_load_nd_dataset<float>(hid_t file_id, const char* dataset_name_,
        int min_dim, int max_dim, Blob<float>* blob) {
  hdf5_load_nd_dataset_helper(file_id, dataset_name_, min_dim, max_dim, blob);
  herr_t status = H5LTread_dataset_float(
    file_id, dataset_name_, blob->mutable_cpu_data());
}

template <>
void hdf5_load_nd_dataset<double>(hid_t file_id, const char* dataset_name_,
        int min_dim, int max_dim, Blob<double>* blob) {
  hdf5_load_nd_dataset_helper(file_id, dataset_name_, min_dim, max_dim, blob);
  herr_t status = H5LTread_dataset_double(
    file_id, dataset_name_, blob->mutable_cpu_data());
}

template <>
void hdf5_save_nd_dataset<float>(
    const hid_t file_id, const string dataset_name, const Blob<float>& blob) {
  hsize_t dims[HDF5_NUM_DIMS];
  dims[0] = blob.num();
  dims[1] = blob.channels();
  dims[2] = blob.height();
  dims[3] = blob.width();
  herr_t status = H5LTmake_dataset_float(
      file_id, dataset_name.c_str(), HDF5_NUM_DIMS, dims, blob.cpu_data());
  CHECK_GE(status, 0) << "Failed to make float dataset " << dataset_name;
}

template <>
void hdf5_save_nd_dataset<double>(
    const hid_t file_id, const string dataset_name, const Blob<double>& blob) {
  hsize_t dims[HDF5_NUM_DIMS];
  dims[0] = blob.num();
  dims[1] = blob.channels();
  dims[2] = blob.height();
  dims[3] = blob.width();
  herr_t status = H5LTmake_dataset_double(
      file_id, dataset_name.c_str(), HDF5_NUM_DIMS, dims, blob.cpu_data());
  CHECK_GE(status, 0) << "Failed to make double dataset " << dataset_name;
}

}  // namespace caffe
