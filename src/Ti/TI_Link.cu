#include "TI_Link.h"

TI_Link::TI_Link(CVX_Link* p)
{
    _link = p;

    strain = p->strain;
}

TI_Link::~TI_Link()
{
}
