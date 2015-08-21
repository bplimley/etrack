//
// ********************************************************************
// * License and Disclaimer                                           *
// *                                                                  *
// * The  Geant4 software  is  copyright of the Copyright Holders  of *
// * the Geant4 Collaboration.  It is provided  under  the terms  and *
// * conditions of the Geant4 Software License,  included in the file *
// * LICENSE and available at  http://cern.ch/geant4/license .  These *
// * include a list of copyright holders.                             *
// *                                                                  *
// * Neither the authors of this software system, nor their employing *
// * institutes,nor the agencies providing financial support for this *
// * work  make  any representation or  warranty, express or implied, *
// * regarding  this  software system or assume any liability for its *
// * use.  Please see the license in the file  LICENSE  and URL above *
// * for the full disclaimer and the limitation of liability.         *
// *                                                                  *
// * This  code  implementation is the result of  the  scientific and *
// * technical work of the GEANT4 collaboration.                      *
// * By using,  copying,  modifying or  distributing the software (or *
// * any work based  on the software)  you  agree  to acknowledge its *
// * use  in  resulting  scientific  publications,  and indicate your *
// * acceptance of all terms of the Geant4 Software license.          *
// ********************************************************************
//
/// \file electromagnetic/TestEm3/include/SteppingAction.hh
/// \brief Definition of the SteppingAction class
//
// $Id$
// 
//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......
//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

#ifndef SteppingAction_h
#define SteppingAction_h 1

#include "G4UserSteppingAction.hh"
#include "globals.hh"
using namespace std;

class DetectorConstruction;
class RunAction;
class EventAction;
#include "G4String.hh"
#include "G4ThreeVector.hh"

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

class SteppingAction : public G4UserSteppingAction
{
  public:
    SteppingAction(DetectorConstruction*, RunAction*, EventAction*);
   ~SteppingAction();

    virtual void UserSteppingAction(const G4Step*);
    // Added Run Manager to Avg. Electron Steps
    //void WriteToOutput(G4SteppingManager* pSM, const G4Step*);
    void WriteToOutput(G4SteppingManager* pSM);
    
    //G4double BirksAttenuation(const G4Step*);
    
  private:
    DetectorConstruction* fDetector;
    RunAction*            fRunAct;
    EventAction*          fEventAct;
    G4String VerboseStepInformation;
  
    G4int NbOfAbsor;
    G4int absorberHit;
    G4int absorNum;
    G4int layerNum;
    G4int Idnow;
    G4int plane;
    G4ThreeVector posA;
    G4double Edep;
    G4double AvgStepEnergy;
    //G4int geHit;
  
  //EventAction*          eventaction;
  //ostringstream outContainer;
  //G4String outFilename;
  //void OpenFile();
  
  
};

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

#endif
