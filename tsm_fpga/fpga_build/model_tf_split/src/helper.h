#include <dirent.h>
#include <vector>
#include <string>

std::vector<std::string> listDir(const std::string& path) {
    std::vector<std::string> res;
    std::string prepend = (path.back() == '/') ? path : path + "/";

    DIR *df;
    struct dirent *file;
    df = opendir(path.c_str());
    if (df) {
        while ((file = readdir(df))) {
            if (!file->d_name || file->d_name[0] == '.')
                continue;
            res.push_back(prepend + file->d_name);
        }
        closedir(df);
    }

    std::sort(res.begin(), res.end());
    return res;
}
