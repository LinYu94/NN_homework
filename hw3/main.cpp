#include <iostream>
#include "svm.h"
#include <stdio.h>
#include <vector>
#include <string.h>
#include <cmath>
using namespace std;

const int CLASSNUM = 12;
const int DIM = 20;

int l;//总数据个数
double* y;//样本的类别向量
svm_node** x;//样本向量数组
int* cln = new int[20];//记录每一个类别的样本数
int* cal = new int[20];//记录前i个类别的样本总数
svm_problem* prob;//用于训练的封装的数据
svm_parameter param;//训练参数
svm_model* svmModel;//训练结果

svm_node** OvRx;//用于训练OvR的封装的数据

double theta = 1;//划分数据时的阈值
int pvpn = 0;
vector<vector<svm_node> > vx;//保存总体样本向量
vector<double> vy;//保存总体样本类别向量

void getOvRx()
{
    OvRx = new svm_node*[l];
    for (int k=0; k<l; k++) {
        OvRx[k] = new svm_node[vx[k].size()];
        for (int a=0; a<vx[k].size(); a++)
            OvRx[k][a] = vx[k][a];
    }
}
void setParam()
{
    param.svm_type = C_SVC;
	param.kernel_type = LINEAR;
	param.degree = 3;
	param.gamma = 0.5;
	param.coef0 = 0;
	param.nu = 0.5;
	param.cache_size = 40;
	param.C = 500;
	param.eps = 1e-3;
	param.p = 0.1;
	param.shrinking = 1;
	// param.probability = 0;
	param.nr_weight = 0;
	param.weight = NULL;
    param.weight_label =NULL;
}


svm_problem* getProb(int i, int j=-1)
{
    svm_problem* spb = new svm_problem;
    if (j == -1){
        spb->l = l;
        spb->y = new double[l];
        spb->x = new svm_node*[l];
        for (int k=0; k<l; k++) {
            spb->x = OvRx;
            if (vy[k] == i)
                spb->y[k] = 1;
            else
                spb->y[k] = -1;
        }
    }
    else {
        spb->l = cln[i]+cln[j];
        spb->x = new svm_node*[spb->l];
        spb->y = new double[spb->l];
        int n=0;
        for (int k=cal[i]; k<cal[i+1]; k++) {
            spb->x[n] = new svm_node[vx[k].size()];
            for (int a=0; a<vx[k].size(); a++)
                spb->x[n][a] = vx[k][a];
            spb->y[n] = i;
            n++;
        }
        for (int k=cal[j]; k<cal[j+1]; k++) {
            spb->x[n] = new svm_node[vx[k].size()];
            for (int a=0; a<vx[k].size(); a++)
                spb->x[n][a] = vx[k][a];
            spb->y[n] = j;
            n++;
        }
    }
    return spb;
}
void read_file(char *filepath)
{
    FILE *fp;
    if((fp=fopen(filepath, "r"))==NULL)
        return ;
    int kind;
    l = 0;
    while(!feof(fp)) {
        fscanf(fp, "%d", &kind);
        cln[kind]++;
        vector<svm_node> feature;
        svm_node tmp;
        for (int i=0; i<DIM; i++) {
            fscanf(fp, "%d:%lf", &tmp.index, &tmp.value);
            if (tmp.value > 10e-6) {
                feature.push_back(tmp);
            }
        }
        tmp.index = -1;
        feature.push_back(tmp);
        //if (kind <= 8 && kind >=7) {
            vx.push_back(feature);
            vy.push_back(kind);
        //}
    }
    cal[0] = 0;
    for (int i=1; i<=CLASSNUM; i++)
        cal[i] = cal[i-1]+cln[i-1];
    l = vx.size();
    cout<<l<<endl;
    fclose(fp);
}
void OvO()
{
    //train
    read_file("train.txt");
    setParam();
    int model_num = CLASSNUM*(CLASSNUM-1)/2;
    int k = 0;
    svm_model** svm_Models = new svm_model*[model_num];

    for (int i=0; i<CLASSNUM; i++) {
        for (int j=i+1; j<CLASSNUM; j++) {
            svm_problem* tmp = getProb(i, j);
            svm_Models[k++] = svm_train(tmp, &param);
        }
    }
    cout<<"OvO训练完毕，正在测试"<<endl;
    //test
    read_file("text.txt");
    int correct_num = 0;
    int* vote = new int[CLASSNUM];
    for (int i=0; i<l; i++) {
        for (int j=0; j<CLASSNUM; j++) vote[j] = 0;
        for (int j=0; j<model_num; j++){
            int res = svm_predict(svm_Models[j], &vx[i][0]);
            vote[res]++;
        }
        int ans = 0;
        int cur = vote[0];
        for (int j=1; j<CLASSNUM; j++) if(vote[j] > cur) {
            ans = j;
            cur = vote[j];
        }
        if (ans == vy[i]) correct_num++;
    }
    cout<<"一共："<<l<<"  正确:"<<correct_num<<endl;
    cout<<"正确率："<<correct_num*1.0/l*100<<"\%"<<endl;
    vx.clear();
    vy.clear();
}

void OvR()
{
    //train
    read_file("train.txt");
    setParam();
    getOvRx();

    //param.svm_type = ONE_CLASS;

    int model_num = CLASSNUM;
    int k = 0;
    svm_model** svm_Models = new svm_model*[model_num];

    for (int i=0; i<CLASSNUM; i++) {
        svm_problem* tmp = getProb(i);
        svm_Models[k++] = svm_train(tmp, &param);
    }
    cout<<"OvR训练完毕，正在测试"<<endl;
    //test
    read_file("text.txt");
    int correct_num = 0;
    double* func = new double[model_num];
    for (int i=0; i<l; i++) {
        for (int j=0; j<model_num; j++){
            double res = 0.0;
            svm_predict_values(svm_Models[j], &vx[i][0], &res);
            func[j] = res;
        }
        int ans = 0;
        double cur = func[0];
        for (int j=1; j<model_num; j++){
            if(func[j] > cur){
                ans = j;
                cur = func[j];
            }
        }
        //cout<<vy[i]<<" "<<ans<<" "<<cur<<endl;
        if (cur >=0 && ans == vy[i])
            correct_num++;
    }
    cout<<"一共："<<l<<"  正确:"<<correct_num<<endl;
    cout<<"正确率："<<correct_num*1.0/l*100<<"\%"<<endl;
    vx.clear();
    vy.clear();
}

void calpart(int a, int b, int part[2]) //划分数据,使得abs(y*a-x*b)+x+y最小
{
    int x, y;
    int ans = abs(a-b);
    x = y = 1;
    for (int i=1; i<=a; i++) {
        for (int j = 1; j<=b; j++) {
            if (abs(j*a-i*b) + theta*(i + j) < ans) {
                x = i;
                y = j;
                ans = abs(j*a-i*b) + theta*(i + j);
            }
        }
    }
    cout<<x<<","<<y<<","<<a/x<<","<<b/y<<","<<ans<<","<<a-a/x*(x-1)<<","<<b-b/y*(y-1)<<endl;

    part[0] = x;
    part[1] = y;
}
svm_model*** MinMax(int a, int b, int part[2])
{
    svm_model*** sm;
    cout<<cln[a]<<"    "<<cln[b]<<endl;
    calpart(cln[a],cln[b],part);
    //构造MinMax
    sm = new svm_model**[part[0]];
    int data_a = cln[a]/part[0];
    int last_a = cln[a] - data_a*(part[0]-1);
    int data_b = cln[b]/part[1];
    int last_b = cln[b] - data_b*(part[1]-1);

    int k_a = cal[a];
    int k_b = cal[b];
    for (int i=0; i<part[0]; i++) {
        sm[i] = new svm_model*[part[1]];
        int an;
        if (i == part[0]-1)
            an = last_a;
        else
            an = data_a;

        int i_b = k_b;
        for (int j=0; j<part[1]; j++) {
            int i_a = k_a;
            svm_problem* tmp = new svm_problem();
            int  bn;
            if (j == part[1]-1)
                bn = last_b;
            else
                bn = data_b;

            tmp->l = an + bn;
            tmp->x = new svm_node*[tmp->l];
            tmp->y = new double[tmp->l];
            int kx = 0;
            for (;kx < an; kx++){
                tmp->x[kx] = &vx[i_a][0];
                tmp->y[kx] = vy[i_a];
                i_a++;
            }
            for (;kx < tmp->l; kx++){
                tmp->x[kx] = &vx[i_b][0];
                tmp->y[kx] = vy[i_b];
                i_b++;
            }
            sm[i][j] = svm_train(tmp, &param);
            pvpn++;
        }

        k_a += an;
    }

    return sm;
}
void PvP()
{
    //train
    read_file("train.txt");
    for (int i=0; i<CLASSNUM; i++) cout<<cln[i]<<" ";
    cout<<endl;
    setParam();
    int model_num = CLASSNUM*(CLASSNUM-1)/2;
    int k = 0;
    svm_model**** svm_Models = new svm_model***[model_num];
    int** data_part = new int*[model_num];
    for (int i=0; i<model_num; i++) data_part[i] = new int[2];

    for (int i=0; i<CLASSNUM; i++) {
        for (int j=i+1; j<CLASSNUM; j++) {
            svm_Models[k] = MinMax(i, j, data_part[k]);
            k++;
        }
    }
    int sss= 0;
    for (int i=0; i<CLASSNUM; i++) {
        for (int j = i+1; j<CLASSNUM; j++) {
            cout<<cln[i]<<","<<cln[j]<<","<<data_part[sss][0]<<","<<data_part[sss][1]<<endl;
            sss++;
        }
    }
    cout<<"PvP训练完毕，一共"<<pvpn<<"个分类器，正在测试"<<endl;


    //test  MinMax 测试
    read_file("text.txt");
    int correct_num = 0;
    int* vote = new int[CLASSNUM];

    for (int i=0; i<l; i++) {
        for (int j=0; j<CLASSNUM; j++) vote[j] = 0;

        for (int j=0; j<model_num; j++){
            //每一个二分类问题使用MinMax
            //cout<<data_part[j][0]<<","<<data_part[j][1]<<endl;
            int* MinRes = new int[data_part[j][0]];
            double* MinValue = new double[data_part[j][0]];
            int k = 0;
            int curMinRes = 1000;
            double curMinValue = 1000;
            for (int a=0; a<data_part[j][0]; a++) {
                for (int b=0; b<data_part[j][1]; b++) {
                    double x = 0.0;
                    int res = svm_predict_values(svm_Models[j][a][b], &vx[i][0], &x);
                    if (res < curMinRes) {
                        curMinRes = res;
                        curMinValue = x;
                    }
                    //cout<<res<<","<<x<<endl;
                    double ssss = 0.3;
                }
                MinRes[k] = curMinRes;
                MinValue[k] = curMinValue;
                k++;
            }
            int curMaxRes = MinRes[0];
            double curMaxValue = MinValue[0];
            for (int a=1; a<data_part[j][0]; a++) {
                if (MinValue[a] > curMaxValue) {
                    curMaxValue = MinValue[a];
                    curMaxRes = MinRes[a];
                }
            }
            vote[curMaxRes]++;
        }


        int ans = 0;
        int cur = vote[0];
        for (int j=1; j<CLASSNUM; j++) if(vote[j] > cur) {
            ans = j;
            cur = vote[j];
        }
        //cout<<vy[i]<<","<<ans<<endl;
        if (ans == vy[i]) correct_num++;
    }
    cout<<"一共："<<l<<"  正确:"<<correct_num<<endl;
    cout<<"正确率："<<correct_num*1.0/l*100<<"\%"<<endl;
    vx.clear();
    vy.clear();
}

void train()
{
    prob = new svm_problem;
    prob->l = l;
    prob->y = new double[l];
    prob->x = new svm_node*[l];
    for (int i=0; i<l; i++) {
        prob->y[i] = vy[i];
        prob->x[i] = new svm_node[vx[i].size()];
        for (int j=0; j<vx[i].size(); j++)
            prob->x[i][j] = vx[i][j];
    }
    setParam();

    svmModel = svm_train(prob, &param);
    vx.clear();
    vy.clear();
}

void test()
{
    double* decValues = new double[CLASSNUM*(CLASSNUM-1)/2];
    cout<<CLASSNUM*(CLASSNUM-1)/2<<endl;
    cout<<"LLLLLL="<<sizeof(decValues)<<endl;
    for (int i=0; i<CLASSNUM*(CLASSNUM-1)/2; i++) decValues[i] = 0.0;
    int correct_num = 0;
    for (int i=0; i<l; i++) {
        double xx = svm_predict_values(svmModel, &vx[i][0], decValues);
        int res = svm_predict(svmModel, &vx[i][0]);
        if (res == vy[i])
            correct_num++;

    }
    cout<<"一共："<<l<<"  正确:"<<correct_num<<endl;
    cout<<"正确率："<<correct_num*1.0/l*100<<"\%"<<endl;
}

int main()
{
    for(int i = 0; i<20; i++){
        cln[i] = 0;
        cal[i] = 0;
    }
    /*用于熟悉libsvm用法
    char* trainFile = "train.txt";
    read_file(trainFile);
    cout<<"各个类别数量："<<endl;
    for (int i=0; i<CLASSNUM; i++) cout<<cln[i]<<" ";
    cout<<endl;

    train();

    char* testFile = "test.txt";
    read_file(testFile);
    test();
    */

    OvO();
    //OvR();
    cout<<"<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"<<endl;
    //PvP();
    return 0;
}
