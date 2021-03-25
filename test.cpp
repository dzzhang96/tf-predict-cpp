#define COMPILER_MSVC
#define NOMINMAX
// #include "test.h"
#include <string>
#include <vector>
#include <fstream>

#include <tensorflow/cc/client/client_session.h>
#include <tensorflow/core/public/session.h>
#include <tensorflow/cc/ops/standard_ops.h>
#include <tensorflow/core/platform/env.h>
#include <tensorflow/cc/ops/image_ops.h>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace tensorflow;
using namespace tensorflow::ops;
using namespace cv;

//void ReadImageNew(const vector<string> &file_names, const int height, const int width, vector<Tensor> &out_tensors) {
//
//	for (auto &file_name : file_names) {
//		auto img = cv::imread(file_name, cv::IMREAD_COLOR);
//		int clen = sizeof(float) * height * width;
//		tensorflow::Tensor input_tensor = tensorflow::Tensor(tensorflow::DT_FLOAT,
//			tensorflow::TensorShape({ 1, 3, height, width }));
//		uchar *p_cls = (uchar *)input_tensor.flat<float>().data();
//
//		cv::Mat iimg, rgb[3];;
//		img.convertTo(iimg, CV_32FC3);
//		split(iimg, &rgb[0]);
//		for (int j = 0; j < 3; j++) {
//			memcpy(p_cls + j * clen, rgb[j].data, clen);
//		}
//		out_tensors.push_back(input_tensor);
//	}
//}

//void SaveImage(const vector<string> &names, const vector<Tensor> &masks)
//{
//	for (size_t i(0); i < names.size(); ++i)
//	{
//		uchar *p = (uchar *)masks[i].flat<float>().data();
//		auto shape = masks[i].shape().dim_sizes();
//
//		int *arr = new int[shape[2] * shape[3]];
//		memcpy(arr, p, shape[2] * shape[3] * sizeof(int));
//
//		cv::Mat image_mat((int)shape[2], (int)shape[3], CV_32F, p);
//		image_mat.convertTo(image_mat, CV_8U, 255, 0);
//		bool flag = cv::imwrite(names[i], image_mat);
//		cout << names[i] << ": " << flag << endl;
//		delete[] arr;
//	}
//}

void ReadImageNew(const vector<string> &file_names, const int height, const int width, vector<Tensor> &out_tensors) {
	if (file_names.size() % 16 != 0) return;//�ж�����ͼƬ�����Ƿ�Ϸ�
	int size_tensor = file_names.size() / 16;//16��ͼƬΪ1�飬�����ж�����
	int clen = sizeof(float) * height * width;//����ÿһ��ͼƬ��1��ͨ��ռ�õ��ֽ���
	//����װ��out_tensors
	for (int i = 0; i < size_tensor; i++) 
	{
		//����tensor
		auto input_tensor = tensorflow::Tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({ 3, 16, height, width,1 }));
		//��ȡtensor����ʼ��ַ
		uchar *p_cls = (uchar *)input_tensor.flat<float>().data();
		//һ���Զ���16��ͼ������imgs��
		vector<Mat> imgs;
		for (int j = 0; j < 16; j++)
		{
			int index = i * 16 + j;//�����ͼƬ·��������ֵ
			string file_name = file_names[index];//��ȡͼƬ·��
			auto img = cv::imread(file_name, cv::IMREAD_COLOR);
			imgs.push_back(img);
		}
		//��ʼ�ڴ濽��
		//3��ͨ��
		for (int j = 0; j < 3; j++) {
			//16��ͼ
			for (int k = 0; k < 16; k++)
			{
				cv::Mat iimg, rgb[3];
				imgs[k].convertTo(iimg, CV_32FC3);
				split(iimg, &rgb[0]);
				//imshow("rgb", rgb[0]);
				//waitKey(0);
				int offset = (j * 16 + k) * clen;//�����ַƫ��
				memcpy(p_cls + offset, rgb[j].data, clen);
			}
		}

		out_tensors.push_back(input_tensor);
	}
}

void SaveImage(const vector<string> &names, const vector<Tensor> &masks)
{
	for (auto &mask : masks)
	{
		uchar *p = (uchar *)mask.flat<float>().data();
		auto shape = mask.shape().dim_sizes();

		int rows = (int)shape[2];
		int cols = (int)shape[3];

		cout << "rows= " << rows << " cols=" << cols << endl;
		for (int i = 0; i < 16; i++)
		{
			int *arr = new int[rows * cols];
			int offset = i * rows * cols * sizeof(int);
			memcpy(arr, p + offset, rows * cols * sizeof(int));

			cv::Mat image_mat(rows, cols, CV_32F, arr);
			image_mat.convertTo(image_mat, CV_8U, 255, 0);

			bool flag = cv::imwrite(names[i], image_mat);
			cout << names[i] << ": " << flag << endl;
			delete[] arr;
		}
	}
}


#include "iostream"

int main(int argc, char **argv)
{
	if (argc < 18)//argc < 3
	{
		cout << "ERROR: Need at least 17 parameters!" << endl;
		return -1;
	}

	// load model
	string model_path(argv[1]);
	std::cout << "modeL_path is " << model_path << std::endl;
	GraphDef graphdef;
	Status load_status = ReadBinaryProto(Env::Default(), model_path, &graphdef);
	if (!load_status.ok())
	{
		cout << "ERROR: Loading model failed! " << model_path << endl;
		cout << load_status.ToString() << endl;
		return -1;
	}
	cout << "INFO: Model loaded." << endl;

	// import model to session
	SessionOptions options;
	unique_ptr<Session> session(NewSession(options));
	Status create_status = session->Create(graphdef);
	if (!create_status.ok())
	{
		cout << "ERROR: Creating graph in session failed! " << endl;
		cout << create_status.ToString() << endl;
		return -1;
	}
	cout << "INFO: Session successfully created." << endl;

	// read image
	vector<string> image_paths;
	for (int i(2); i < argc; ++i)
		image_paths.push_back(string(argv[i]));
	vector<Tensor> image_tensors;
	//ReadImageNew(image_paths, 128, 128, image_tensors);
	ReadImageNew(image_paths, 256, 256, image_tensors);
	vector<Tensor> outputs;
	vector<Tensor> masks;
	for (auto &image_tensor : image_tensors)
	{
		std::vector<std::pair<string, Tensor> > inputs;

		inputs.push_back({ "Placeholder", image_tensor });

		tensorflow::Tensor placeholder = tensorflow::Tensor(tensorflow::DT_FLOAT, TensorShape());
		placeholder.scalar<float>()() = true;
		inputs.push_back({ "Placeholder_4", placeholder });

		//inputs.push_back({ "Input", image_tensor });

		//tensorflow::Tensor drop_out = tensorflow::Tensor(tensorflow::DT_FLOAT, TensorShape());
		//drop_out.scalar<float>()() = 1;
		//inputs.push_back({ "DropOut", drop_out });

		//tensorflow::Tensor phase = tensorflow::Tensor(tensorflow::DT_BOOL, TensorShape());
		//phase.scalar<bool>()() = true;
		//inputs.push_back({ "Phase", phase });

		std::cout << image_tensor.shape() << endl;

		TF_CHECK_OK(session->Run(inputs, { "output/Sigmoid" }, {}, &outputs));
		//TF_CHECK_OK(session->Run(inputs, { "output" }, {}, &outputs));
		std::cout << "outputs size is " << outputs.size() << std::endl;

		masks.push_back(outputs[0]);
		std::cout << outputs[0].shape() << std::endl;
		outputs.clear();
	}
	cout << "INFO: Run inference finished." << endl;
	std::cout << masks[0].shape() << std::endl;
	vector<string> out_paths;
	for (auto &path : image_paths)
		out_paths.push_back(string(path.begin(), path.end() - 3) + "JPEG");

	SaveImage(out_paths, masks);
}