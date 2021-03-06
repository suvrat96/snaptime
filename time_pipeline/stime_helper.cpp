#include "stime_helper.hpp"

//includes directories. returns the full path
void TTimeFFile::GetAllFiles(TStr& Path, TStrV& FnV, bool OnlyDirs){
  DIR* dir = opendir(Path.CStr());
  AssertR(dir != NULL, "Directory could not be opened");
  struct dirent *dir_entity = readdir(dir);
  while (dir_entity != NULL) {
  	if (strcmp(dir_entity->d_name, "..") != 0 && strcmp(dir_entity->d_name, ".") != 0) {
  		TStr dirname = Path + TStr("/") + TStr(dir_entity->d_name);
  		if (!OnlyDirs || TDir::Exists(dirname)) FnV.Add(dirname);
  	}
    dir_entity = readdir(dir);
  }
  closedir(dir);
}

TVec<TStr> TCSVParse::readCSVLine(std::string line, char delim, bool TrimWs) {
    TVec<TStr> vec_line;
    std::istringstream is(line);
    std::string temp;
    while(getline(is, temp, delim)) {
      std::string val = temp;
      if(TrimWs) {
        val = trim(val);
      }
      vec_line.Add(TStr(val.c_str()));
    }
    return vec_line;
}

std::string TCSVParse::trim(std::string const& str)
{
    if(str.empty())
        return str;
    std::size_t firstScan = str.find_first_not_of(' ');
    std::size_t first = (firstScan == std::string::npos) ? str.length() : firstScan;
    std::size_t last = str.find_last_not_of(' ');
    return str.substr(first, last-first+1);
}

//fname is based on primary and secondary hash of ids
// primHash_secHash (does not include .bin)
TStr TCSVParse::CreateIDVFileName(const TTIdVec & IdVec) {
    TStr prim_hash = TInt::GetHexStr(IdVec.GetPrimHashCd()); //dirnames are based on hash of ids
    TStr sec_hash = TInt::GetHexStr(IdVec.GetSecHashCd()); //dirnames are based on hash of ids
    TStr result = prim_hash + TStr("_") + sec_hash;
    return result;
}