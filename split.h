#include <sstream>
#include <vector>

template<typename T>
void split(const std::string &s, char delim, T result)
{
    std::stringstream ss;
    ss.str(s);
    std::string item;
    while(std::getline(ss, item, delim))
    {
        *(result++) = item;
    }
}

std::vector<std::string> split(const std::string &s, char delim)
{
    std::vector<std::string> elems;
    split(s, delim, std::back_inserter(elems));
    return elems;
}

std::vector<std::string> operator|(std::string s, char c)
{
    return split(s, c);
}
