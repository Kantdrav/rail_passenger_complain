import express from "express";
import { voiceHandler,voiceResponse,callStatusHandler ,submitComplaint} from "../api/test.js";

const router = express.Router();

// POST /api/complaints
router.post('/voice-handler/:complaintId', voiceHandler);
router.post('/voice-response/:complaintId',voiceResponse);
router.post('/call-status/:complaintId', callStatusHandler);
router.post("/complaint", submitComplaint);


export default router;

 
