#if !defined(TI_LINK_H)
#define TI_LINK_H

#include "VX_Link.h"

class TI_Link
{
private:
    /* data */
public:
    TI_Link(CVX_Link* p);
    ~TI_Link();

    CVX_Link* _link;
	float strain;

};



#endif // TI_LINK_H
