#include "fillData.hpp"
#include "Snap.h"

Eigen::MatrixXd FillData::createAndFillData(std::string filename,long long initialTimestamp,int duration){
    CreateData dt = CreateData();
    TFIn FIn (TStr(filename.c_str()));
    dt.Load(FIn);
    //dt.parseData(filename);
    return dt.fillData(initialTimestamp,duration);
}
