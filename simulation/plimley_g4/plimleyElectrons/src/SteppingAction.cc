/// \file electromagnetic/TestEm3/src/SteppingAction.cc
/// \brief Implementation of the SteppingAction class
//
// $Id$
//
//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......
//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

#include "SteppingAction.hh"

#include "DetectorConstruction.hh"
#include "RunAction.hh"
#include "EventAction.hh"
#include "HistoManager.hh"

#include "G4Step.hh"
#include "G4Positron.hh"
#include "CCDRunManager.hh"
#include "G4PhysicalConstants.hh"
// TM
using namespace std;
#include "G4String.hh"
#include "G4VProcess.hh"
#include "G4UnitsTable.hh"

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

SteppingAction::SteppingAction(DetectorConstruction* det, RunAction* run,
                               EventAction* evt)
:G4UserSteppingAction(),fDetector(det),fRunAct(run),fEventAct(evt) 
{ }

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

SteppingAction::~SteppingAction()
{}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

void SteppingAction::UserSteppingAction(const G4Step* aStep)
{
  
  //track informations
  const G4StepPoint* prePoint = aStep->GetPreStepPoint();   
  const G4StepPoint* endPoint = aStep->GetPostStepPoint();
  const G4ParticleDefinition* particle = aStep->GetTrack()->GetDefinition();
  
  // Event Flags
  absorberHit = 0;
  //TM: Tracking Matrix
  G4SteppingManager* pSM = fpSteppingManager;
  // Check Particle Charge (neutral particles will be gamma in these simulations)
  G4int Chg = pSM->GetTrack()->GetDefinition()->GetPDGCharge();

  
  // IF Photon (temp hold the photon tracking info)

  G4VPhysicalVolume* volume = prePoint->GetTouchableHandle()->GetVolume();
  NbOfAbsor = fDetector->GetNbOfAbsor(); // Number of Absorbers
  // Check for Absorber Volume
  for (G4int k=1; k<=NbOfAbsor; k++) {
    if (volume == fDetector->GetAbsorber(k)){
      //eventaction->AddAbs(edep,stepl);
      // Flag CCD Interation - Record Electrons
      absorberHit = 1;
      
    }
    // Any Layer should count as absorber Hit at this time.
    //else
    //{
    //  absorberHit = 0;
    //}
  }
  
  // Values of interest (For both Summary info and for Tracking Matrix
  if (absorberHit==1)
  {

    //here we are in an absorber. Locate it
    //
    absorNum  = prePoint->GetTouchableHandle()->GetCopyNumber(0);
    layerNum  = prePoint->GetTouchableHandle()->GetCopyNumber(1);
    
    // unique identificator of layer+absorber
    Idnow = (fDetector->GetNbOfAbsor())*layerNum + absorNum;
    //G4int plane;
  }
  else{
    Idnow = 0;
  }
  
  //if (volume != fDetector->GetCalor()) return;
  //if sum of absorbers do not fill exactly a layer: check material, not volume.
  //G4Material* mat = volume->GetLogicalVolume()->GetMaterial();
  //if (mat == fDetector->GetWorldMaterial()) return;
  
  // Check Material for The Source Volume and for the Cryobox, 
  //if (mat == fDetector->GetSourceMaterial()) return;
  //if (mat == fDetector->GetCryoMaterial()) return;
  //if (mat == fDetector->GetCryoCutoutMaterial()) return;
  
  // Save Tracking Matrix Information
  // Records all stepping information of possible interest
  //if ((Chg == 0) ||(volume == fDetector->GetCalor()))
  
  
  if ((Chg == 0) ||(absorberHit==1)){
    WriteToOutput(pSM);
    
  }
  
  
  // Basic Information Gathering within the Absorber (CCD):
  //
  // Return at this point if no Absorber Hit
  if (absorberHit==0) return;
  
  // collect energy deposit taking into account track weight
  G4double edep = aStep->GetTotalEnergyDeposit()*aStep->GetTrack()->GetWeight();
  
  // collect step length of charged particles
  G4double stepl = 0.;
  if (particle->GetPDGCharge() != 0.) {
    stepl = aStep->GetStepLength();
    fRunAct->AddChargedStep();
  } else { fRunAct->AddNeutralStep(); }
  
  //  G4cout << "Nabs= " << absorNum << "   edep(keV)= " << edep << G4endl;
  
  // sum up per event
  fEventAct->SumEnergy(absorNum,edep,stepl);
  
  //longitudinal profile of edep per absorber
  if (edep>0.) {
    G4AnalysisManager::Instance()->FillH1(MaxAbsor+absorNum, 
					  G4double(layerNum+1), edep);
  }
 //*/
  //energy flow
  //

  //
  //leaving the absorber ?
  if (endPoint->GetStepStatus() == fGeomBoundary) {
    G4ThreeVector position  = endPoint->GetPosition();
    G4ThreeVector direction = endPoint->GetMomentumDirection();
    G4double sizeYZ = 0.5*fDetector->GetCalorSizeYZ();       
    G4double Eflow = endPoint->GetKineticEnergy();
    if (particle == G4Positron::Positron()) Eflow += 2*electron_mass_c2;
    if ((std::abs(position.y()) >= sizeYZ) || (std::abs(position.z()) >= sizeYZ)) 
                                  fRunAct->SumLateralEleak(Idnow, Eflow);
    else if (direction.x() >= 0.) fRunAct->SumEnergyFlow(plane=Idnow+1, Eflow);
    else                          fRunAct->SumEnergyFlow(plane=Idnow,  -Eflow);
    
  }   
  //*/
////  example of Birk attenuation
///G4double destep   = aStep->GetTotalEnergyDeposit();
///G4double response = BirksAttenuation(aStep);
///G4cout << " Destep: " << destep/keV << " keV"
///       << " response after Birks: " << response/keV << " keV" << G4endl;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

//void SteppingAction::WriteToOutput(G4SteppingManager* pSM, const G4Step* aStep)
void SteppingAction::WriteToOutput(G4SteppingManager* pSM)
{
  //	G4int writeTestFlag = 0;
  // Precision Settings
  G4int prec = 10;        // Precision in Postions [mm]
  G4int prec2 = 8;        // Precision in StepLength and totalTrackLength   [um]
  // Precision in Energy   [eV]
  //	CCD Interation Flag
    // G4int numberESteps = 10; // Plan to print every step w/in Calor
  //G4int absorberHit = 0;
  
  //G4cout << "WriteToOutput(pSM)" << G4endl;
  
  G4int ID = pSM->GetTrack()->GetTrackID();
  G4int PID = pSM->GetTrack()->GetParentID();
  G4int stepNum = pSM->GetTrack()->GetCurrentStepNumber();
  G4double Chg = pSM->GetTrack()->GetDefinition()->GetPDGCharge();
  G4ThreeVector pos0 = pSM->GetStep()->GetPreStepPoint()->GetPosition();
  G4ThreeVector pos1 = pSM->GetStep()->GetPostStepPoint()->GetPosition();
  G4double trackL = pSM->GetTrack()->GetTrackLength();
  G4double stepL = pSM->GetStep()->GetStepLength();
  G4double E = pSM->GetTrack()->GetKineticEnergy();
  //G4double DE = pSM->GetStep()->GetDeltaEnergy();
  
  Edep = pSM->GetStep()->GetTotalEnergyDeposit();
  //G4double deltaE;
  G4bool secondaryCreationStepFlag = 0; // Printing Secondaries not tallying them
  G4int numberESteps = 1;
  
  CCDRunManager* pRM = (CCDRunManager*)CCDRunManager::GetRunManager();
  
  // Get List of Secondaries and find total energy exiting
  // Note DE_offset will be positive
  G4TrackVector* fSecondary = pSM->GetfSecondary();
  G4int TotSecondaries = pSM->GetfN2ndariesAlongStepDoIt() +
  pSM->GetfN2ndariesAtRestDoIt() + pSM->GetfN2ndariesPostStepDoIt();
  
  G4double DE_offset = 0;
  if(TotSecondaries>0){
	for(size_t lp1=(*fSecondary).size()-TotSecondaries; lp1<(*fSecondary).size(); lp1++)
	{
      DE_offset += (*fSecondary)[lp1]->GetKineticEnergy();
	}
    if (DE_offset > 100*eV){          //this 100 eV is kind of arbitrary
      secondaryCreationStepFlag = 1;
    }
  }
  
  //pRM->AccumulateAvgStepEnergy(Edep);
  //pRM->AccumulateDEOffset(DE_offset);
  // Only Avg. Electron steps Within the Absorber
  if (absorberHit==1 && Chg==-1)
  {
    if (stepNum==1)
    {
      //pRM->SetTempEnergy(E-DE);
      pRM->SetTempPosition(pos0);
      pRM->ResetAvgStepEnergy(Edep);
      //pRM->ResetDEOffset(DE_offset);
    }
    else
    {
        pRM->AccumulateAvgStepEnergy(Edep);
        //pRM->AccumulateDEOffset(DE_offset);
        if (stepNum%numberESteps==0 || secondaryCreationStepFlag || E==0)
        {
          //deltaE = E-pRM->GetTempEnergy()+pRM->GetDEOffset();
          AvgStepEnergy = pRM->GetAvgStepTempEnergy();
        
          pRM->ResetAvgStepEnergy(0);
          posA = pRM->GetTempPosition();
          pRM->SetTempPosition(pos0);
        }
      }
    }
  else
  {
    posA = pos0;
    AvgStepEnergy = Edep; // For Photon steps inside absorber
  }
  // Check Volume - Set Event Tracking Flags (stepNum restricts electrons tracked to primary electrons only... (not quite right... ) (2/14/2014, switched back to no stepNum==1.
  //if ((absorberHit == 1) && (Edep > 0) && (stepNum==1) && (Chg ==-1))	fEventAct->ChangeAbsorberHitFlag();
  if ((absorberHit == 1) && (Edep > 0) && (Chg ==-1))	fEventAct->ChangeAbsorberHitFlag(); // 2/14/14 //Amy says I can change this
  
  // Second Detector
  //
  //if (volume == detector->GetGe() && PID==0 && E==0)		fEventAct->ChangeGeHitFlag();

  
  
  //      Old Etracking Gamma-Sim (pre-2012)
  //	if (((stepNum==1||stepNum%10==0||E==0)&&(Chg ==-1 )) || ((ID==1)&&(PID==0)&&(stepNum==1))||((PID==0)&&Edep>0))
  /*// Old E Track Code (2012)
  if ( (	(absorberHit==1)
        &&(stepNum==1||stepNum%numberESteps==0||secondaryCreationStepFlag==1||E==0)
        &&(Chg ==-1 ))
      || (Chg==0 && ((stepNum==1 || DE<0) || (PID==0))))
   */
  /*// New E Track Matrix (Nov. 2013)
    if ( (	(absorberHit==1)                                // Absorber
          //&& (stepNum==1 || Edep>0 || E==0))
          &&((Chg == -1 )|| Edep>0))
          //&&((Chg == -1 ) // All Electron steps only...
          //|| (Chg==0 && ((stepNum==1 || DE<0) || (PID==0)))) // Photon
        || (Chg==0 && ((stepNum==1 || Edep>0) || (PID==0)))) // Photon
   */
   // New E Track Matrix (Feb. 2014) (Should cover everything... but should be simplified at some point)
  if (
        ((absorberHit==1)                                // Absorber
            &&(Chg ==-1 )       // Electron Tracks
            &&(stepNum==1||stepNum%numberESteps==0||secondaryCreationStepFlag==1||E==0)
         )
        //|| (absorberHit==1 && Chg==0 && Edep>0) // Photons in the Absorber
        || (PID==0)   //(stepNum==1 && PID==0)  // First Event
        || (Chg==0 && (stepNum==1 || Edep>0))   // Photon History
        )
    
  {
	// New Verbose Step Tracking Method:
    VerboseStepInformation="";
    
    ostringstream VerboseStepInformationStream;
    VerboseStepInformationStream.str("");
    VerboseStepInformationStream    
    << ID << ","                // col 1
    << PID << ","               // col 2
    << stepNum << ","           // col 3
    << Chg << ","               // col 4
    << setprecision (prec)      
    << pos0.getX()/mm << ","    // col 5
    << pos0.getY()/mm << ","    // col 6
    << pos0.getZ()/mm << ","    // col 7
    << pos1.getX()/mm << ","    // col 8
    << pos1.getY()/mm << ","    // col 9
    << pos1.getZ()/mm << ","    // col 10
    << setprecision (prec2)
    << trackL/um << ","         // col 11
    << stepL/um << ","          // col 12
    //<< (E-DE)/eV << "," // Initial Energy (obsolet in next Geant4 version)
    << E/eV << ","              // col 13 - Final Energy (good)
    //<< DE/eV << ","     // Delta Energy (obsolet in next Geant4 version)
    << Edep/eV <<  ","          // col 14 - Total Energy Deposited (Geant4) On current Step
    << AvgStepEnergy/eV //<< ","  // col 15 - Sum of 'numberESteps' Steps Geant4 "Total Energy Deposited"
    //<< Idnow                    // col 16 - layer ID
    << G4endl;
    
    //G4cout << VerboseStepInformationStream.str() << G4endl;
    
	VerboseStepInformation = VerboseStepInformationStream.str();
    
	fEventAct->AddVerboseStepInformation(VerboseStepInformation);
  }
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......
/*
G4double SteppingAction::BirksAttenuation(const G4Step* aStep)
{
  //Example of Birk attenuation law in organic scintillators.
  //adapted from Geant3 PHYS337. See MIN 80 (1970) 239-244
  //
  G4Material* material = aStep->GetTrack()->GetMaterial();
  G4double birk1       = material->GetIonisation()->GetBirksConstant();
  G4double destep      = aStep->GetTotalEnergyDeposit();
  G4double stepl       = aStep->GetStepLength();
  G4double charge      = aStep->GetTrack()->GetDefinition()->GetPDGCharge();
  //
  G4double response = destep;
  if (birk1*destep*stepl*charge != 0.)
  {
    response = destep/(1. + birk1*destep/stepl);
  }
  return response;
}
*/


//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

