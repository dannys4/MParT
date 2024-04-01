#include "MParT/Utilities/Miscellaneous.h"
#include <iostream>

using namespace mpart;

std::string mpart::GetOption(std::unordered_map<std::string,std::string> const& map, 
                             std::string                                 const& key, 
                             std::string                                 const& defaultValue)
{

    // Extract the polynomial type
    std::string output;
    auto it = map.find(key);
    if(it==map.end()){
        output = defaultValue;
    }else{
        output = map.at(key);
    }
    return output;
}

void mpart::DeprecationWarning(std::string const& deprecated, std::string const& replacement, int const& max_warn)
{
    static std::unordered_map<std::string,int> warnings;
    if(warnings.find(deprecated)==warnings.end()){
        warnings[deprecated] = 0;
    }
    if(warnings[deprecated]<max_warn){
        std::cerr << "\033[33mWARNING: " << deprecated << " is deprecated.  Please use " << replacement << " instead.\033[0m" << std::endl;
        warnings[deprecated]++;
    }
}