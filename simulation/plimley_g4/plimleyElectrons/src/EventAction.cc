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
/// \file electromagnetic/TestEm3/src/EventAction.cc
/// \brief Implementation of the EventAction class
//
// $Id$
//
//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......
//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

#include "EventAction.hh"

#include "RunAction.hh"
#include "EventActionMessenger.hh"
#include "HistoManager.hh"

#include "G4Event.hh"

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

EventAction::EventAction(DetectorConstruction* det, RunAction* run)
:fDetector(det), fRunAct(run)
{
  fDrawFlag = "none";
  fPrintModulo = 10000;
  fEventMessenger = new EventActionMessenger(this);
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

EventAction::~EventAction()
{
  delete fEventMessenger;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

void EventAction::BeginOfEventAction(const G4Event* evt)
{   
  G4int evtNb = evt->GetEventID();

  //survey printing
  if (evtNb%fPrintModulo == 0)
    G4cout << "\n---> Begin Of Event: " << evtNb << G4endl;
    
  //initialize EnergyDeposit per event
  //
  for (G4int k=0; k<MaxAbsor; k++) {
    fEnergyDeposit[k] = fTrackLengthCh[k] = 0.0;   
  }

  //initialize Detector Hit Flags
  AbsorberHitFlag = false;
  //GeHitFlag = false;
  VerboseEventInformationStream="";
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

void EventAction::EndOfEventAction(const G4Event*)
{
  for (G4int k=1; k<=fDetector->GetNbOfAbsor(); k++) {
     fRunAct->FillPerEvent(k,fEnergyDeposit[k],fTrackLengthCh[k]);                       
     if (fEnergyDeposit[k] > 0.)
             G4AnalysisManager::Instance()->FillH1(k, fEnergyDeposit[k]);
  }

  // Coincident Events Only:
  // if (AbsorberHitFlag == true && GeHitFlag == true){
  // CCD Response only:
  if (AbsorberHitFlag == true){
    //G4cout << "AbsorberHitFlag==true in EventAction" << G4endl;
    fRunAct->AddVerboseEventInformation(VerboseEventInformationStream);
  }
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

void EventAction::AddVerboseStepInformation(G4String VerboseStepInformation)
{
  //G4cout << VerboseStepInformation  << G4endl;
  VerboseEventInformationStream.append(VerboseStepInformation);
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......


