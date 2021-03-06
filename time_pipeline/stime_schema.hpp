#ifndef STIME_SCHEMA_H
#define STIME_SCHEMA_H

enum TColType {TIME, SENSOR, ID, NO_ID};


class TTSchema {
public:
	//Note: must be exactly one Time value and at least one sensor value

	/*
	 * Vector of ID types in order as they will be placed into the binaries
	 * Example: <"Driver", "MachineID", "SENSOR">
	 */
	TVec<TStr> IdNames;
	/*
	 * Map of ID names to index placed in above vector
	 */
	THash<TStr, TInt> IDName_To_Index;

	/*
	 * Vector specifying directory behavior
	 * Each value is a directory that one needs to dive into
	 * If the value is NULL then the dir name should not be an ID. Otherwise, the dir name
	 * should be associated with the given ID name
	 * If the value is "TIME" then the dir name is used as time
	 * must have at least one value (default is NULL)
	 */
	TVec<TStr> Dirs;

	/*
	 * Vector specifying the columns of the file
	 * If the value type is TIME then the value should be used as a time value
	 * If the value type is NO_ID then the value should be ignored
	 * If the value type is SENSOR then the value should be used as a Sensor Value
	 * If the value type is ID then the value should be used as an ID val for the given
	 * ID name
	 *
	 * Example:
	 * < ("TimeData", TIME), ("BogusID", NO_ID), ("Car Model", ID), ("Acceleration", SENSOR)>
	 */
	 TVec<TPair<TStr, TColType> > FileSchema;
	 // For each behavior, have the indices of the relevant column numbers
	 TVec<TVec<TInt> > FileSchemaIndexList;

	 /*
	  * hash from sensor name to type. If not in the hash, assume TFlt
	  */
	 THash<TStr, TType> SensorType;
	 TType defaultType; /* by default, float */

	char FileDelimiter; /* default ',' */

	 /*
	  * Time manipulation
	  */
	 TBool IsTimeStr; // true if the time needs to be formatted (ie not a long)
	 TStr TimeFormatter; // the format string to convert the time into a tm
	 TBool HasTime; //time column is specified

public:
	TTSchema() : FileSchemaIndexList(4) {
		IsTimeStr = false;
		HasTime = false;
		defaultType = FLOAT;
		FileDelimiter = ',';
	}

	void ReadSchemaFile(TStr FileName);
	TTime ConvertTime(TStr timeVal) const;
	void PrintSchema() const;


private:
	void AddIDName(TStr & IdName);
	static bool GetNextSchemaLine(std::ifstream & infile, std::string & line);
	void ReadTimeConversion(std::ifstream & infile);
	void ReadFileHierarchy(std::ifstream & infile);
	void ReadDataSchema(std::ifstream & infile);
	void ReadSchemaTypes(std::ifstream & infile);
	void divideFileSchemaByType();

};
#endif
