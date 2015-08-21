#ifndef CCDRunManager_h
#define CCDRunManager_h 1

#include "G4RunManager.hh"
#include "G4ThreeVector.hh"

class CCDRunManager : public G4RunManager 
{
  public:
      
       CCDRunManager();
      ~CCDRunManager() {;};
	void SetTempEnergy(G4double);
    void SetAvgStepTempEnergy(G4double);
	void SetTempPosition(G4ThreeVector);
  
	G4double GetTempEnergy();
  	G4double GetDEOffset();
    G4double GetAvgStepTempEnergy();
	G4ThreeVector GetTempPosition();

	void AccumulateDEOffset(G4double);
    void AccumulateAvgStepEnergy(G4double);
    void ResetDEOffset(G4double);
    void ResetAvgStepEnergy(G4double);
private:
	G4double TempEnergy;
    G4double AvgStepTempEnergy;
	G4ThreeVector TempPosition;
	G4double DE_Offset;

};

#endif

