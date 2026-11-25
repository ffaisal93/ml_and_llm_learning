#include <algorithm>
#include <cassert>
#include <functional>
#include <iostream>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>
using HANDLE = int;
struct Trade { std::string symbol; int size; std::string flag; };

// Contract: one calling thread; callbacks execute synchronously and may subscribe/unsubscribe.
// Empty symbol/flag and size==0 are wildcards; other fields match exactly, combined with AND.
class TradeProcessor {
    struct Subscription { Trade filter; std::function<void(const Trade&)> callback; };
    std::map<HANDLE,Subscription> subscriptions;
    HANDLE next_handle = 1;
    static bool matches(const Trade& t, const Trade& f) {
        return (f.symbol.empty() || t.symbol==f.symbol)
            && (f.size==0 || t.size==f.size)
            && (f.flag.empty() || t.flag==f.flag);
    }
public:
    HANDLE subscribe(const Trade& filter, std::function<void(const Trade&)> callback) {
        if (!callback || filter.size<0) throw std::invalid_argument("Invalid subscription");
        if(next_handle==std::numeric_limits<HANDLE>::max())
            throw std::overflow_error("Handle space exhausted");
        HANDLE handle=next_handle++;
        subscriptions.emplace(handle,Subscription{filter,std::move(callback)});
        return handle;
    }
    void unsubscribe(HANDLE handle) { subscriptions.erase(handle); }
    void onNewTrade(const Trade& trade) {
        std::vector<HANDLE> handles;
        for(const auto& pair: subscriptions) handles.push_back(pair.first);
        for(HANDLE handle: handles) {
            auto found=subscriptions.find(handle);
            if(found==subscriptions.end() || !matches(trade,found->second.filter)) continue;
            // Copy before calling: callback may erase its own map entry.
            auto callback=found->second.callback;
            try { callback(trade); }
            catch(const std::exception& error) { std::cerr<<"Callback failed: "<<error.what()<<'\n'; }
            catch(...) { std::cerr<<"Callback failed with nonstandard exception\n"; }
        }
    }
};

int main() {
    TradeProcessor processor;
    int exact=0, any=0;
    auto first=processor.subscribe({"AAPL",100,"X"},[&](const Trade&){++exact;});
    processor.subscribe({"",0,""},[&](const Trade&){++any;});
    processor.onNewTrade({"AAPL",100,"X"});
    processor.onNewTrade({"AAPL",101,"X"});
    assert(exact==1 && any==2);
    processor.unsubscribe(first); processor.unsubscribe(first);
    processor.onNewTrade({"AAPL",100,"X"}); assert(exact==1 && any==3);
    TradeProcessor reentrant; HANDLE later=0;
    reentrant.subscribe({"",0,""},[&](const Trade&){reentrant.unsubscribe(later);});
    later=reentrant.subscribe({"",0,""},[&](const Trade&){assert(false);});
    reentrant.onNewTrade({"X",1,""});
    TradeProcessor self; HANDLE own=0; int count=0;
    own=self.subscribe({"",0,""},[&](const Trade&){++count;self.unsubscribe(own);});
    self.onNewTrade({"X",1,""}); self.onNewTrade({"X",1,""});assert(count==1);
    std::cout<<"W539 filter, unsubscribe and reentrancy checks passed\n";
}
