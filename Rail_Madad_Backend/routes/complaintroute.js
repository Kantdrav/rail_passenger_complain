import express from "express";
import { submitComplaint,statusupdate,getAllcomplaint} from "../api/complaint.js";

const router = express.Router();

// POST /api/complaints
router.post("/complaint", submitComplaint);
router.get("/getcomplaint",getAllcomplaint);

router.put("/complaint/:complaintId/resolved", statusupdate);
export default router;
