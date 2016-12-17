#include <iostream>
#include <map>
#include <regex>
#include <sstream>
#include <set>

using namespace std;
using ll = long long;

int main() {
	regex header("Benchmark.*cpu.*size.*Mode.*Cnt.*Score.*Error.*Units");
	string line;
	getline(cin, line);
	if(!regex_match(line, header)) {
		cerr << "not the right format!\n";
		return 1;
	}
	set<string> categories;
	map<ll, map<string, string>> table;
	while(getline(cin, line)) {
		stringstream ss(line);
		string _;
		getline(ss, _, '.'); //get rid of BenchmarkClass
		string name, cpu, score;
		ll size;
		ss >> name >> cpu >> size >> _ >> _ >> score;
		if(cpu == "false")
			name += "_gpu";
		categories.insert(name);
		table[size][name] = score;
	}
	cout << "size";
	for(const string &head : categories)
		cout << " " << head;
	cout << endl;
	for(const auto &e: table) {
		cout << e.first;
		for(const string &cat : categories) {
			cout << " ";
			auto it = e.second.find(cat);
			if(it == end(e.second))
				cout << "nan";
			else
				cout << it->second;
		}
		cout << endl;
	}
}
