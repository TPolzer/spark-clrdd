#include <fstream>
#include <iostream>
#include <cstdlib>
#include <unistd.h>
#include <cassert>

using namespace std;

int main(int argc, char** argv) {
	char buffer[1024];
	ssize_t error = readlink("/proc/self/exe", buffer, sizeof(buffer));
	assert(error != -1);
	string self_path(buffer);
	for(int i=0; i<self_path.size(); ++i) {
		if(self_path[self_path.size() - i - 1] == '/') {
			self_path.resize(self_path.size() - i);
			break;
		}
	}
	string gradlew_path = self_path + "gradlew";
	execv(gradlew_path.c_str(), argv);
	return 1;
}
