#include <caffe/caffe.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <algorithm>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <time.h>

using namespace caffe;
using namespace std;
using namespace cv;

int main(int argc, char* argv[])
{
    string img = "../testing/test_images/testFace.jpg";
    if(argc != 2){
        cout << "Usage: " << argv[0] << " inImage"<< endl;
    }else{
        img = argv[1];
    }

    string network = "../vanilla-40/model_68p/deploy.prototxt";
    string weights = "../vanilla-40/model_68p/_iter_1400000.caffemodel"; //landmark.caffemodel";
    Net<float> *net = new Net<float>(network,TEST);
    int input_w = 40, input_h = 40;

    net->CopyTrainedLayersFrom(weights);

    Caffe::set_mode(Caffe::CPU);


    //开始检测，返回一系列的边界框
    clock_t start;
    int cnt = 0;

    //for(int i = 0;i < dets.size();i++)
    {
        Mat srcROI = imread(img); //(image, Rect(tmp.left(),tmp.top(),tmp.right()-tmp.left(),tmp.bottom() - tmp.top()));

        Mat img2;
        cvtColor(srcROI,img2,CV_RGB2GRAY);

        img2.convertTo(img2, CV_32FC1);
        Size dsize = Size(input_w, input_h);
        Mat img3 = Mat(dsize, CV_32FC1);
        resize(img2, img3, dsize, 0,0,INTER_CUBIC);
        cv::imwrite("image3.jpg", img3);

        Mat tmp_m, tmp_sd;
        double m = 0, sd = 0;
        meanStdDev(img3, tmp_m, tmp_sd);
        m = tmp_m.at<double>(0,0);
        sd = tmp_sd.at<double>(0,0);

        img3 = (img3 - m)/(0.000001 + sd);

        if (img3.channels() * img3.rows * img3.cols != net->input_blobs()[0]->count())
            LOG(FATAL) << "Incorrect " << img3<< ", resize to correct dimensions.\n";
        // prepare data into array
        float *data = (float*)malloc( img3.rows * img3.cols * sizeof(float));

        int pix_count = 0;

        for (int i = 0; i < img3.rows; ++i) {
            for (int j = 0; j < img3.cols; ++j) {
                float pix = img3.at<float>(i, j);
                float* p = (float*)(data);
                p[pix_count] = pix;
                ++pix_count;
            }
        }

        std::vector<Blob<float>*> in_blobs = net->input_blobs();
        in_blobs[0]->Reshape(1, 1, img3.rows, img3.cols);
        net->Reshape();

        in_blobs[0]->set_cpu_data((float*)data);
        Timer total_timer;
        total_timer.Start();

        net->Forward();
        cout << " total time = " << total_timer.MicroSeconds() / 1000 <<endl;
        const boost::shared_ptr<caffe::Blob<float> > feature_blob = net->blob_by_name("Dense2");//获取该层特征

        float feat_dim = feature_blob->count() / feature_blob->num();//计算特征维度
        cout << feat_dim << endl;
        const float* data_ptr = (const float *)feature_blob->cpu_data();//特征块数据


        std::vector<float> feat2;

        for (int i = 0; i < feat_dim; i++)
        {
            feat2.push_back(*data_ptr);
            if (i < feat_dim - 1)
                data_ptr++;
        }

        for(int i = 0;i < feat_dim/2;i++)
        {
            cout << "--- [x,y] " << feat2[2*i] <<" " << feat2[2*i + 1] << endl;
            Point x = Point(int(feat2[2*i]*(srcROI.cols)),int(feat2[2*i + 1]*(srcROI.rows)));
            cv::circle(srcROI, x, 0.05, Scalar(0, 0, 255), 4, 8, 0);
        }
        imwrite("result.jpg", srcROI);
        free(data);
    }
    free(net);

    return 0;

}
