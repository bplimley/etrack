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
/// \file electromagnetic/TestEm3/include/EventAction.hh
/// \brief Definition of the EventAction class
//
// $Id$
//
//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......
//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

#ifndef EventAction_h
#define EventAction_h 1

#include "G4UserEventAction.hh"
#include "globals.hh"
#include "DetectorConstruction.hh"

class RunAction;
class EventActionMessenger;

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

class EventAction : public G4UserEventAction
{
  public:  
    EventAction(DetectorConstruction*, RunAction*);
   ~EventAction();

    virtual void BeginOfEventAction(const G4Event*);
    virtual void   EndOfEventAction(const G4Event*);
    
    void SetDrawFlag   (G4String val)  {fDrawFlag    = val;};
    void SetPrintModulo(G4int    val)  {fPrintModulo = val;};
    
    void ChangeAbsorberHitFlag() {AbsorberHitFlag = true;};
    //void ChangeGeHitFlag() {GeHitFlag = true;};
  
    // New Event Tracking Code
    void AddVerboseStepInformation(G4String VerboseStepInformation);
    
    void SumEnergy(G4int k, G4double de, G4double dl)
        {fEnergyDeposit[k] += de; fTrackLengthCh[k] += dl;};          
        
  private:  
    DetectorConstruction* fDetector;
    RunAction*            fRunAct;
    
    G4double              fEnergyDeposit[MaxAbsor];
    G4double              fTrackLengthCh[MaxAbsor];

    G4bool                AbsorberHitFlag;
    //G4bool GeHitFlag;
    G4String VerboseEventInformationStream;	// Temp Holds tracked events
        
    G4String              fDrawFlag; 
    G4int                 fPrintModulo;         
    EventActionMessenger* fEventMessenger;
};

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

#endif

    
