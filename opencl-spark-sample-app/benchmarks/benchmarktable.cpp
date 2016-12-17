#include <iostream>
#include <map>
#include <regex>
#include <sstream>
#include <set>
#include <iomanip>

using namespace std;
using ll = long long;

int main() {
	cout << fixed << setprecision(3);
	regex header("Benchmark.*cpu.*size.*Mode.*Cnt.*Score.*Error.*Units");
	string line;
	getline(cin, line);
	if(!regex_match(line, header)) {
		cerr << "not the right format!\n";
		return 1;
	}
	set<string> categories;
	map<ll, map<string, pair<ll, double>>> table;
	while(getline(cin, line)) {
		if(regex_match(line, header))
			continue;
		stringstream ss(line);
		string _;
		getline(ss, _, '.'); //get rid of BenchmarkClass
		string name, cpu;
		double score;
		ll size, count;
		ss >> name >> cpu >> size >> _ >> count >> score;
		if(cpu == "false")
			name += "_gpu";
		categories.insert(name);
		auto old = table[size][name];
		ll ncount = old.first + count;
		double nscore = (old.second * old.first + score * count) / ncount;
		table[size][name] = make_pair(ncount, nscore);
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
				cout << it->second.second;
		}
		cout << endl;
	}
}
