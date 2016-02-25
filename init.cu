#include "luaT.h"
#include "THC.h"

#include "utils.cu"

#include "TripletCriterion.cu"

LUA_EXTERNC DLL_EXPORT int luaopen_libtriplet(lua_State *L);

int luaopen_libtriplet(lua_State *L)
{
  lua_newtable(L);
  triplet_TripletCriterion_init(L);

  return 1;
}
