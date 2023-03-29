#include "HTCTrackingSys.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <stdint.h>
#include <fstream>
#include <iostream>
#include <sstream>


HTCTrackingSys::HTCTrackingSys(bool useEyeTracking=false, bool useHandTracking=false) {
	this->useEyeTracking = useEyeTracking;
	this->useHandTracking = useHandTracking;
}



/// @brief Starts VR system
void HTCTrackingSys::start() {
	// starts eye tracking
	if (useEyeTracking) {
		shouldShutDownEyeTracking = false;
		if (!ViveSR::anipal::Eye::IsViveProEye()) {
			fprintf(stderr, "[HTCTrackingSys] This HMD does not support eye-tracking!\n");
			exit(EXIT_FAILURE);
		}
		int anipalError = ViveSR::anipal::Initial(ViveSR::anipal::Eye::ANIPAL_TYPE_EYE, NULL);
		switch (anipalError) {
			case ViveSR::Error::WORK:
				break;
			case ViveSR::Error::RUNTIME_NOT_FOUND:
				fprintf(stderr, "[HTCTrackingSys] SRAnipal runtime not installed!\n");
				exit(EXIT_FAILURE);
			default:
				fprintf(stderr, "[HTCTrackingSys] Failed to initialize SRAnipal!\n");
				exit(EXIT_FAILURE);
		}
		// Launch a thread to poll data from the SRAnipal SDK
		// We poll data asynchronously so as to not slow down the VR rendering loop
		eyeTrackingThread = new std::thread(&HTCTrackingSys::pollAnipal, this);
	}
	// starts hand tracking
	if (useHandTracking) {
	}
}



/// @brief Stops and cleans up VR system
void HTCTrackingSys::stop() {
	// stop eye tracking thread
	if (useEyeTracking) {
		shouldShutDownEyeTracking = true;
		eyeTrackingThread->join();
	}
	// stop hand tracking thread
	if (useHandTracking) {
	}
}



/// @brief Returns whether the current VR system supports eye tracking
/// @return whether eye tracking is supported
bool HTCTrackingSys::hasEyeTrackingSupport() {
	return ViveSR::anipal::Eye::IsViveProEye();
}



/// @brief Queries eye tracking data and returns to user
/// @return list containing eye tracking data
py::list HTCTrackingSys::getEyeTrackingData() {
	py::list eyeData, combinedOrigin, combinedDirection, leftEyeOrigin, leftEyeDirection, rightEyeOrigin, rightEyeDirection;
	
	combinedOrigin.append(eyeTrackingData.combinedOrigin.z);
	combinedOrigin.append(eyeTrackingData.combinedOrigin.x);
	combinedOrigin.append(eyeTrackingData.combinedOrigin.y);
	combinedDirection.append(eyeTrackingData.combinedDirection.z);
	combinedDirection.append(eyeTrackingData.combinedDirection.x);
	combinedDirection.append(eyeTrackingData.combinedDirection.y);

	leftEyeOrigin.append(eyeTrackingData.leftEyeOrigin.z);
	leftEyeOrigin.append(eyeTrackingData.leftEyeOrigin.x);
	leftEyeOrigin.append(eyeTrackingData.leftEyeOrigin.y);
	leftEyeDirection.append(eyeTrackingData.leftEyeDirection.z);
	leftEyeDirection.append(eyeTrackingData.leftEyeDirection.x);
	leftEyeDirection.append(eyeTrackingData.leftEyeDirection.y);

	rightEyeOrigin.append(eyeTrackingData.rightEyeOrigin.z);
	rightEyeOrigin.append(eyeTrackingData.rightEyeOrigin.x);
	rightEyeOrigin.append(eyeTrackingData.rightEyeOrigin.y);
	rightEyeDirection.append(eyeTrackingData.rightEyeDirection.z);
	rightEyeDirection.append(eyeTrackingData.rightEyeDirection.x);
	rightEyeDirection.append(eyeTrackingData.rightEyeDirection.y);
	
	// Set validity to false if eye tracking is not being used
	if (useEyeTracking) {
		eyeData.append(eyeTrackingData.isCombinedPoseValid);
		eyeData.append(eyeTrackingData.isLeftPoseValid);
		eyeData.append(eyeTrackingData.isRightPoseValid);
	}
	else {
		eyeData.append(false);
		eyeData.append(false);
		eyeData.append(false);
	}
	eyeData.append(combinedOrigin);
	eyeData.append(combinedDirection);
	eyeData.append(leftEyeOrigin);
	eyeData.append(leftEyeDirection);
	eyeData.append(rightEyeOrigin);
	eyeData.append(rightEyeDirection);
	eyeData.append(eyeTrackingData.leftPupilDiameter);
	eyeData.append(eyeTrackingData.rightPupilDiameter);
	eyeData.append(eyeTrackingData.leftEyeOpenness);
	eyeData.append(eyeTrackingData.rightEyeOpenness);
	return eyeData;
}



/// @brief Polls SRAnipal to get updated eye tracking information
void HTCTrackingSys::pollAnipal() {
	while (!this->shouldShutDownEyeTracking) {
		if (ViveSR::anipal::Eye::GetEyeData(&this->eyeData) == ViveSR::Error::WORK) { 
			// Record pupil measurements
			eyeTrackingData.leftPupilDiameter = -1;
			eyeTrackingData.rightPupilDiameter = -1;
			if (ViveSR::anipal::Eye::DecodeBitMask(this->eyeData.verbose_data.left.eye_data_validata_bit_mask,
				ViveSR::anipal::Eye::SINGLE_EYE_DATA_PUPIL_DIAMETER_VALIDITY))
				eyeTrackingData.leftPupilDiameter = this->eyeData.verbose_data.left.pupil_diameter_mm;
			if (ViveSR::anipal::Eye::DecodeBitMask(this->eyeData.verbose_data.right.eye_data_validata_bit_mask,
				ViveSR::anipal::Eye::SINGLE_EYE_DATA_PUPIL_DIAMETER_VALIDITY))
				eyeTrackingData.rightPupilDiameter = this->eyeData.verbose_data.right.pupil_diameter_mm; 
			// Record eye openness
			eyeTrackingData.leftEyeOpenness = -1;
			eyeTrackingData.rightEyeOpenness = -1;
			if (ViveSR::anipal::Eye::DecodeBitMask(this->eyeData.verbose_data.left.eye_data_validata_bit_mask,
				ViveSR::anipal::Eye::SINGLE_EYE_DATA_EYE_OPENNESS_VALIDITY))
				eyeTrackingData.leftEyeOpenness = this->eyeData.verbose_data.left.eye_openness;
			if (ViveSR::anipal::Eye::DecodeBitMask(this->eyeData.verbose_data.right.eye_data_validata_bit_mask,
				ViveSR::anipal::Eye::SINGLE_EYE_DATA_EYE_OPENNESS_VALIDITY))
				eyeTrackingData.rightEyeOpenness = this->eyeData.verbose_data.right.eye_openness; 

			// Both origin and dir are relative to the HMD coordinate system, so we need to transform them into HMD coordinate system
			eyeTrackingData.isCombinedPoseValid = false;
			eyeTrackingData.isLeftPoseValid = false;
			eyeTrackingData.isRightPoseValid = false;
			// combined eye origin and direction
			int isCombinedOriginValid = ViveSR::anipal::Eye::DecodeBitMask(this->eyeData.verbose_data.combined.eye_data.eye_data_validata_bit_mask,
				ViveSR::anipal::Eye::SINGLE_EYE_DATA_GAZE_DIRECTION_VALIDITY);
			int isCombinedDirValid = ViveSR::anipal::Eye::DecodeBitMask(this->eyeData.verbose_data.combined.eye_data.eye_data_validata_bit_mask,
				ViveSR::anipal::Eye::SINGLE_EYE_DATA_GAZE_ORIGIN_VALIDITY);
			if (isCombinedOriginValid && isCombinedDirValid) {
				eyeTrackingData.isCombinedPoseValid = true;
				// Returns value in mm, so need to divide by 1000 to get meters (OG uses meters)
				auto combinedGazeOrigin = this->eyeData.verbose_data.combined.eye_data.gaze_origin_mm;
				glm::vec3 combinedEyeSpaceOrigin(combinedGazeOrigin.x / 1000.0f, combinedGazeOrigin.y / 1000.0f, combinedGazeOrigin.z / 1000.0f);
				eyeTrackingData.combinedOrigin = combinedEyeSpaceOrigin;
				auto combinedGazeDirection = this->eyeData.verbose_data.combined.eye_data.gaze_direction_normalized;
				glm::vec3 combinedEyeSpaceDir(combinedGazeDirection.x, combinedGazeDirection.y, combinedGazeDirection.z);
				eyeTrackingData.combinedDirection = combinedEyeSpaceDir;
			}

			// left eye origin and direction
			int isLeftOriginValid = ViveSR::anipal::Eye::DecodeBitMask(this->eyeData.verbose_data.left.eye_data_validata_bit_mask,
				ViveSR::anipal::Eye::SINGLE_EYE_DATA_GAZE_DIRECTION_VALIDITY);
			int isLeftDirValid = ViveSR::anipal::Eye::DecodeBitMask(this->eyeData.verbose_data.left.eye_data_validata_bit_mask,
				ViveSR::anipal::Eye::SINGLE_EYE_DATA_GAZE_ORIGIN_VALIDITY);
			if (isLeftOriginValid && isLeftDirValid) {
				eyeTrackingData.isLeftPoseValid = true;
				// Returns value in mm, so need to divide by 1000 to get meters (OG uses meters)
				auto leftGazeOrigin = this->eyeData.verbose_data.left.gaze_origin_mm;
				glm::vec3 leftEyeSpaceOrigin(leftGazeOrigin.x / 1000.0f, leftGazeOrigin.y / 1000.0f, leftGazeOrigin.z / 1000.0f);
				eyeTrackingData.leftEyeOrigin = leftEyeSpaceOrigin;
				auto leftGazeDirection = this->eyeData.verbose_data.left.gaze_direction_normalized;
				glm::vec3 leftEyeSpaceDir(leftGazeDirection.x, leftGazeDirection.y, leftGazeDirection.z);
				eyeTrackingData.leftEyeDirection = leftEyeSpaceDir;
			}

			// right eye origin and direction
			int isRightOriginValid = ViveSR::anipal::Eye::DecodeBitMask(this->eyeData.verbose_data.right.eye_data_validata_bit_mask,
				ViveSR::anipal::Eye::SINGLE_EYE_DATA_GAZE_DIRECTION_VALIDITY);
			int isRightDirValid = ViveSR::anipal::Eye::DecodeBitMask(this->eyeData.verbose_data.right.eye_data_validata_bit_mask,
				ViveSR::anipal::Eye::SINGLE_EYE_DATA_GAZE_ORIGIN_VALIDITY);
			if (isRightOriginValid && isRightDirValid) {
				eyeTrackingData.isRightPoseValid = true;
				// Returns value in mm, so need to divide by 1000 to get meters (OG uses meters)
				auto rightGazeOrigin = this->eyeData.verbose_data.right.gaze_origin_mm;
				glm::vec3 rightEyeSpaceOrigin(rightGazeOrigin.x / 1000.0f, rightGazeOrigin.y / 1000.0f, rightGazeOrigin.z / 1000.0f);
				eyeTrackingData.rightEyeOrigin = rightEyeSpaceOrigin;
				auto rightGazeDirection = this->eyeData.verbose_data.right.gaze_direction_normalized;
				glm::vec3 rightEyeSpaceDir(rightGazeDirection.x, rightGazeDirection.y, rightGazeDirection.z);
				eyeTrackingData.rightEyeDirection = rightEyeSpaceDir;
			}
		}
	}
}



PYBIND11_MODULE(HTCTrackingSys, m) {
	py::class_<HTCTrackingSys> pymodule = py::class_<HTCTrackingSys>(m, "HTCTrackingSys");
	pymodule.def(py::init<bool, bool>());
	pymodule.def("start", 					&HTCTrackingSys::start);
	pymodule.def("stop", 					&HTCTrackingSys::stop);
	pymodule.def("getEyeTrackingData", 		&HTCTrackingSys::getEyeTrackingData);
	pymodule.def("hasEyeTrackingSupport", 	&HTCTrackingSys::hasEyeTrackingSupport);

	#ifdef VERSION_INFO
		m.attr("__version__") = VERSION_INFO;
	#else
		m.attr("__version__") = "dev";
	#endif
}