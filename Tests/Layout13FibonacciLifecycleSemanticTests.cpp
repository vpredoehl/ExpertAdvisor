#include "FeatureAblation.hpp"
#include "FeatureLayout.hpp"
#include "MarketStructureRegistry.hpp"
#include "ModelInputContract.hpp"
#include "ModelInputExpansion.hpp"
#include <cassert>
#include <cstddef>
#include <string>
#include <vector>
template<class F> bool bad(F f){try{f();}catch(const std::invalid_argument&){return true;}return false;}
int main(){
 static_assert(causal_price_level_raw_feature_size==123);
 static_assert(causal_fibonacci_lifecycle_feature_size==167);
 static_assert(feature_size==167);
 static_assert(EA::kCausalPriceLevelRawModelInputWidth==127);
 static_assert(EA::kCausalFibonacciLifecycleModelInputWidth==171);
 static_assert(EA::kCurrentModelInputWidth==171);
 static_assert(EA::kModelInputSemanticLayoutVersion==13);
 assert(fibLifecycleUp0382ReachedCountLogCol==123);
 assert(fibLifecycleDownACloseBeyondYoungestAgeLogCol==166);
 EA::MarketStructure::ValidateRegistry();
 assert(EA::MarketStructure::FindFamily("fibonacci_lifecycle")!=nullptr);
 assert(EA::MarketStructure::ResolvePrefix("fibonacci_lifecycle",12).empty());
 auto ch=EA::MarketStructure::ResolvePrefix("fibonacci_lifecycle",13);
 assert(ch.size()==44);
 for(std::size_t i=0;i<ch.size();++i){assert(ch[i]->tensorColumn==123+i);assert(ch[i]->introducedSemanticLayout==13);}
 assert(bad([]{(void)EA::FeatureAblationMask::Resolve("fibonacci_lifecycle.*",12);}));
 auto m=EA::FeatureAblationMask::Resolve("fibonacci_lifecycle.*",13).resolvedMask;
 assert(m.tensorColumns().size()==44);
 assert(m.CanonicalText()==EA::kCausalFibonacciLifecycleAblationMaskText);
 assert(EA::FeatureAblationMask::Parse("fibonacci_lifecycle.*").CanonicalText()==m.CanonicalText());
 std::string c=m.CanonicalText();
 assert(c.find("youngest_reach_age_log1p")!=std::string::npos);
 assert(c.find("youngest_directional_close_age_log1p")!=std::string::npos);
 assert(c.find("reached_youngest_age_log")==std::string::npos);
 auto exact=EA::MarketStructure::FindChannel("fibonacci_lifecycle.up.0500.youngest_reach_age_log1p");
 assert(exact&&exact->persistedFeatureId=="fib_lifecycle_up_0500_youngest_reach_age_log1p");
 auto ct=EA::ResolveModelInputContract(EA::kCurrentModelInputWidth,feature_size);
 assert(ct.tensorFeatureCount==167);
 std::vector<float> a(feature_size,1),o(EA::kCurrentModelInputWidth,-1);
 EA::CopyTensorFeaturesForModelInput(o.data(),a.data(),ct,m);
 for(std::size_t i=0;i<123;++i)assert(o[i]==1);
 for(std::size_t i=123;i<167;++i)assert(o[i]==0);
 auto plan=EA::BuildRegisteredInputWidthExpansionPlan(EA::kCausalPriceLevelRawModelInputWidth,EA::kCausalFibonacciLifecycleModelInputWidth,EA::kRegisteredModelInputWidths,EA::kAppendedTensorFeatureSemantics);
 assert(plan.sourceTensorFeatureCount==123&&plan.expandedTensorFeatureCount==167);
 assert(plan.newlyIntroducedTensorFeatures.size()==44);
 assert(plan.newlyIntroducedTensorFeatures.front()=="fib_lifecycle_up_0382_reached_count_log");
 assert(plan.newlyIntroducedTensorFeatures.back()=="fib_lifecycle_down_youngest_a_close_beyond_age_log1p");
}
