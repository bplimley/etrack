#include "CCDRunManager.hh"

CCDRunManager::CCDRunManager()
{
    TempEnergy = 0.0;
    AvgStepTempEnergy = 0.0;
}

void CCDRunManager::SetTempEnergy(G4double E)
{
  TempEnergy = E;
}

void CCDRunManager::SetAvgStepTempEnergy(G4double E)
{
	AvgStepTempEnergy = E;
}

void CCDRunManager::SetTempPosition(G4ThreeVector pos)
{
	TempPosition = pos;
}

G4double CCDRunManager::GetTempEnergy()
{
	return TempEnergy;
}

G4double CCDRunManager::GetDEOffset()
{
  return DE_Offset;
}

G4double CCDRunManager::GetAvgStepTempEnergy()
{
  return AvgStepTempEnergy;
}

G4ThreeVector CCDRunManager::GetTempPosition()
{
	return TempPosition;
}

void CCDRunManager::AccumulateDEOffset(G4double DE)
{
  DE_Offset += DE;
}

void CCDRunManager::AccumulateAvgStepEnergy(G4double AvgStepTempEnergy)
{
  AvgStepTempEnergy += AvgStepTempEnergy;
}

void CCDRunManager::ResetDEOffset(G4double RE)
{
  DE_Offset = RE;
}

void CCDRunManager::ResetAvgStepEnergy(G4double RE)
{
  AvgStepTempEnergy = RE;
}

