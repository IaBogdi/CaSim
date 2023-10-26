#include "IonChannel.h"

Point* IonChannel::GetCoordinates()
{
    return coordinates.get();
}

KineticModel* IonChannel::GetKineticModel()
{
    return model.get();
}
