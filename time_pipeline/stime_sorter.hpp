/*
 * Container Class for methods to perform sorting of bucketed data (phase II)
 */
class TSTimeSorter {
private:
	class RawDataCmp {
	private:
		TCmp<TTime> cmp;
	public:
		int operator() (const TTRawData& x, const TTRawData& y) const {
			return cmp(x.Val1, y.Val1);
		}
	};
public:
	static void SortBucketedDataDir(TStr DirPath, bool ClearData, TTSchema* schema_p);

private:
	template<class TVal>
	static void WriteSortedData(TStr DirPath, TTIdVec& IDs, TTRawDataV& SortedData, TVal (*val_convert)(TStr),
		bool ClearData) {
		// convert all strings into actual data types
		TSTime<TVal> result(IDs);
		for (int i=0; i < SortedData.Len(); i++) {
			TTime ts = SortedData[i].GetVal1();
			TVal val = val_convert(SortedData[i].GetVal2());
			TPair<TTime, TVal> new_val(ts, val);
			result.TimeData.Add(new_val);
		}
		TStr OutFile = DirPath + TStr("/") + TCSVParse::CreateIDVFileName(IDs) + TStr(".out");
		if (ClearData) {
			std::cout << "clearing directory: " << DirPath.CStr() << std::endl;
			TStrV FnV;
			TFFile::GetFNmV(DirPath, TStrV::GetV("bin"), false, FnV);
			for (int i=0; i<FnV.Len(); i++) {
				std::cout << FnV[i].CStr() << std::endl;
				TFile::Del(FnV[i]);
			}
		}

		//save file out
		TFOut outstream(OutFile);
		result.Save(outstream);
		//TODO, put granularity
	}

};