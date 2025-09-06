import nodemailer from "nodemailer";
import jwt from "jsonwebtoken";
import express from "express";
import bcrypt from "bcryptjs";
import twilio from "twilio";
import Complaint from "../schemas/complaintschema.js";
import Staff from "../schemas/staffschema.js";
// import { VoiceResponse } from 'twilio';
const { VoiceResponse } = twilio;

// Initialize Twilio client with proper error handling
let twilioClient;
let twilioNumber;

try {
  const accountSid = process.env.TWILIO_ACCOUNT_SID;
  const authToken = process.env.TWILIO_AUTH_TOKEN;
  twilioNumber = process.env.TWILIO_NUMBER; // Fixed: Should be string with +

  if (!accountSid || !authToken || !twilioNumber) {
    console.error("Twilio credentials missing");
  } else {
    twilioClient = twilio(accountSid, authToken);
    console.log("✅ Twilio client initialized successfully");
  }
} catch (error) {
  console.error("❌ Twilio initialization error:", error);
}
// Updated submitComplaint function with coach number handling

export const submitComplaint = async (req, res) => {
  try {
    console.log("twilioClient status:", twilioClient ? "✅ INITIALIZED" : "❌ NOT INITIALIZED");
    console.log("TWILIO SID:", process.env.TWILIO_ACCOUNT_SID);
console.log("TWILIO TOKEN:", process.env.TWILIO_AUTH_TOKEN);

const client = twilio(
  process.env.TWILIO_ACCOUNT_SID,
  process.env.TWILIO_AUTH_TOKEN
);

client.api.accounts(process.env.TWILIO_ACCOUNT_SID)
  .fetch()
  .then(acc => console.log("Auth ✅", acc.friendlyName))
  .catch(err => console.error("Auth ❌", err));
    // CHANGE 1: Add coachNumber to destructuring
    const { category, subCategory, details, trainNo, coachNo, pnrNumber } = req.body;
    
    // Validate required fields
    if (!category || !subCategory || !trainNo) {
      return res.status(400).json({
        success: false,
        message: "Category, subcategory, and train number are required"
      });
    }

    // CHANGE 2: Add coachNumber validation for specific categories
    // if (['Coach & Seat', 'Cleanliness', 'AC & Temperature'].includes(category) && !coachNumber) {
    //   return res.status(400).json({
    //     success: false,
    //     message: "Coach number is required for this complaint category"
    //   });
    // }

    // Category to Hindi mapping for voice call
    const categoryHindi = {
      "Coach & Seat": "डिब्बा और सीट",
      "Cleanliness": "सफाई",
      "Food & Catering": "खाना और कैटरिंग", 
      "AC & Temperature": "एयर कंडीशनर और तापमान",
      "Electrical": "बिजली",
      "Staff Behavior": "स्टाफ का व्यवहार",
      "Luggage Issues": "सामान की समस्या",
      "Ticketing": "टिकट की समस्या",
      "Safety & Security": "सुरक्षा",
      "Punctuality": "समय की पाबंदी",
      "Facilities": "सुविधाएं",
      "Accessibility": "पहुंच की सुविधा"
    };

    // Department mapping
    const departmentMap = {
      "Coach & Seat": "Engineering",
      "Cleanliness": "Cleaning",
      "Food & Catering": "Catering",
      "AC & Temperature": "Engineering", 
      "Electrical": "Engineering",
      "Staff Behavior": "HR",
      "Luggage Issues": "Operations",
      "Ticketing": "Ticketing",
      "Safety & Security": "Security",
      "Punctuality": "Operations",
      "Facilities": "Facilities",
      "Accessibility": "SpecialAssistance",
    };

    const department = departmentMap[category] || "General";

    // CHANGE 3: Save complaint with coachNumber
    const complaint = new Complaint({
      category,
      subCategory,
      details,
      trainNo,
      coachNo, // Add this field
      pnrNumber,
      department,
      status: "Pending",
      callAttempts: 0,
      acknowledgmentReceived: false,
      maxCallAttempts: 3, // Add max attempts limit
      createdAt: new Date()
    });

    await complaint.save();

    // Find staff
    let staff = await Staff.findOne({ department, trainNo });
    console.log(`Staff found for train ${trainNo}:`, staff);
    
    if (!staff) {
      staff = await Staff.findOne({ department });
      console.log(`No staff found for train ${trainNo}, using general department staff:`, staff);
    }

    if (staff && twilioClient) {
      // CHANGE 4: Pass coachNumber to IVR call
      await initiateIVRCall(complaint._id, staff, category, categoryHindi, subCategory, details, coachNo);
    } else {
      console.log(`❌ No staff found for department: ${department} or Twilio not configured`);
      complaint.status = "Pending - No staff assigned";
      await complaint.save();
    }

    res.status(201).json({
      success: true,
      complaint,
      message: staff ? "Complaint submitted and staff notification initiated" : "Complaint submitted but no staff assigned"
    });
  } catch (error) {
    console.error("Complaint Submit Error:", error);
    res.status(500).json({ success: false, message: "Server Error" });
  }
};
async function initiateIVRCall(complaintId, staff, category, categoryHindi, subCategory, details, coachNo) {
  try {
    const hindiCategory = categoryHindi[category] || category;
    
    // Include coach number in complaint details
    let complaintDetails = '';
    if (details) complaintDetails += `Vivaran: ${details}. `;
    if (coachNo) complaintDetails += `Coach Number: ${coachNo}. `;
    
    // Use proper API routes
    const twimlUrl = `https://rail-madad-backend-p4vg.onrender.com/api/test/voice-handler/${complaintId}?category=${encodeURIComponent(hindiCategory)}&subcategory=${encodeURIComponent(subCategory)}&details=${encodeURIComponent(complaintDetails)}&coach=${encodeURIComponent(coachNo || '')}`;
    
    const call = await twilioClient.calls.create({
      to: staff.mobileNumber,
      from: twilioNumber,
      url: twimlUrl,
      statusCallback: `https://rail-madad-backend-p4vg.onrender.com/api/test/call-status/${complaintId}`,
      statusCallbackEvent: ['answered', 'completed', 'no-answer', 'busy', 'failed'],
      timeout: 30,
      record: false
    });

    console.log("✅ IVR call initiated:", call.sid);
    
    // Update complaint with call info
    await Complaint.findByIdAndUpdate(complaintId, {
      $set: {
        staffCalled: true,
        callSid: call.sid,
        assignedTo: {
          name: staff.name,
          phone: staff.mobileNumber,
          department: staff.department
        },
        callAttempts: 1,
        lastCallTime: new Date()
      },
      $push: {
        callHistory: {
          attempt: 1,
          callSid: call.sid,
          status: 'initiated',
          timestamp: new Date()
        }
      }
    });
  } catch (err) {
    console.error("❌ IVR Call Error:", err);
    await fallbackToSMS(complaintId, staff, categoryHindi, subCategory, details, coachNo);
  }
}
// Update voiceHandler function
export const voiceHandler = async (req, res) => {
  try {
    const { complaintId } = req.params;
    const { category, subcategory, details, coach } = req.query;
    
    // Check if complaint is already acknowledged
    const complaint = await Complaint.findById(complaintId);
    if (complaint && complaint.acknowledgmentReceived) {
      const twiml = new twilio.twiml.VoiceResponse();
      twiml.say({
        voice: 'alice',
        language: 'en-IN'
      }, 'This complaint has already been acknowledged. Thank you.');
      
      res.type('text/xml');
      return res.send(twiml.toString());
    }
    
    const twiml = new twilio.twiml.VoiceResponse();
    
    // Create IVR menu with options
    const gather = twiml.gather({
      numDigits: 1,
      action: `https://rail-madad-backend-p4vg.onrender.com/api/test/voice-response/${complaintId}`,
      method: 'POST',
      timeout: 10
    });
    
    // Include coach number in voice message
    let voiceMessage = `Namaskar. Train complaint notification. Category: ${category}. Problem: ${subcategory}.`;
    if (coach) voiceMessage += ` Coach Number: ${coach}.`;
    if (details) voiceMessage += ` ${details}`;
    voiceMessage += ' Press 1 to hear again, Press 2 to confirm receipt and resolve complaint.';
    
    gather.say({
      voice: 'alice',
      language: 'en-IN'
    }, voiceMessage);
    
    // If no input, retry the menu
    twiml.redirect(`https://rail-madad-backend-p4vg.onrender.com/api/test/voice-handler/${complaintId}?category=${encodeURIComponent(category)}&subcategory=${encodeURIComponent(subcategory)}&details=${encodeURIComponent(details)}&coach=${encodeURIComponent(coach || '')}`);
    
    res.type('text/xml');
    res.send(twiml.toString());
  } catch (error) {
    console.error("Voice Handler Error:", error);
    res.status(500).send("Internal Server Error");
  }
};
// CHANGE 9: Improved Call Status Handler with proper retry logic
export const callStatusHandler = async (req, res) => {
  try {
    const { complaintId } = req.params;
    const { CallStatus, CallSid } = req.body;
    console.log("Twilio status callback body:", req.body);

    console.log(`Call status update for complaint ${complaintId}: ${CallStatus}`);
    
    const complaint = await Complaint.findById(complaintId);
    if (!complaint) {
      return res.status(404).send("Complaint not found");
    }

    // CHANGE 10: Don't retry if already acknowledged
    if (complaint.acknowledgmentReceived) {
      console.log(`✅ Complaint ${complaintId} already acknowledged. No further action needed.`);
      return res.status(200).send("OK - Already acknowledged");
    }

    // Update call status and history
    await Complaint.findByIdAndUpdate(complaintId, {
      $set: {
        lastCallStatus: CallStatus,
        lastCallStatusTime: new Date()
      },
      $push: {
        callHistory: {
          attempt: complaint.callAttempts,
          callSid: CallSid,
          status: CallStatus,
          timestamp: new Date()
        }
      }
    });

    // CHANGE 11: Handle different call statuses with improved logic
    if (['no-answer', 'busy', 'failed'].includes(CallStatus)) {
      console.log(`Call not answered/failed for complaint ${complaintId}. Current attempts: ${complaint.callAttempts}`);
      
      if (complaint.callAttempts < complaint.maxCallAttempts) {
        // Schedule retry after 2 minutes
        console.log(`Scheduling retry ${complaint.callAttempts + 1} for complaint ${complaintId} in 2 minutes`);
        setTimeout(() => {
          retryCall(complaintId);
        }, 120000); // 2 minutes
      } else {
        // Max attempts reached, send SMS and update status
        console.log(`Max call attempts (${complaint.maxCallAttempts}) reached for complaint ${complaintId}. Sending SMS fallback...`);
        
        const staff = await Staff.findOne({ 
          department: complaint.department, 
          mobileNumber: complaint.assignedTo.phone 
        });
        
        if (staff) {
          await fallbackToSMS(complaintId, staff, "शिकायत", complaint.subCategory, complaint.details, complaint.coachNo);
        }
        
        // Update complaint status
        await Complaint.findByIdAndUpdate(complaintId, {
          $set: {
            status: "Call Failed - SMS Sent",
            callsFailed: true
          }
        });
      }
    } else if (CallStatus === 'completed') {
      // Call was completed, but check if it was acknowledged
      console.log(`Call completed for complaint ${complaintId}. Checking acknowledgment status...`);
      
      // Wait 30 seconds then check if acknowledged
      setTimeout(async () => {
        const updatedComplaint = await Complaint.findById(complaintId);
        if (updatedComplaint && !updatedComplaint.acknowledgmentReceived) {
          console.log(`Call completed but not acknowledged for complaint ${complaintId}. May need retry.`);
          
          if (updatedComplaint.callAttempts < updatedComplaint.maxCallAttempts) {
            // Schedule another call
            setTimeout(() => {
              retryCall(complaintId);
            }, 60000); // 5 minutes wait
          }
        }
      }, 30000); // Wait 30 seconds
    }

    res.status(200).send("OK");
  } catch (error) {
    console.error("Call Status Handler Error:", error);
    res.status(500).send("Internal Server Error");
  }
};

export const voiceResponse = async (req, res) => {
  try {
    const { complaintId } = req.params;
    const digits = req.body.Digits;
    
    const twiml = new twilio.twiml.VoiceResponse();

    if (digits === '1') {
      // Repeat the message
      twiml.redirect(`https://rail-madad-backend-p4vg.onrender.com/api/test/voice-handler/${complaintId}`);
    } else if (digits === '2') {
      // Check if already acknowledged
      const complaint = await Complaint.findById(complaintId);
      
      if (complaint && complaint.acknowledgmentReceived) {
        twiml.say({
          voice: 'alice',
          language: 'en-IN'
        }, 'This complaint has already been acknowledged. Thank you.');
      } else {
        // Confirm acknowledgment
        twiml.say({
          voice: 'alice',
          language: 'en-IN'
        }, 'Dhanyawad. Complaint acknowledgment received. Please take necessary action.');
        
        // Update database
        await Complaint.findByIdAndUpdate(complaintId, {
          $set: {
            acknowledgmentReceived: true,
            acknowledgedAt: new Date(),
            status: "Acknowledged",
            acknowledgedBy: "Staff via IVR"
          }
        });
        
        console.log(`✅ Complaint ${complaintId} acknowledged by staff via IVR`);
      }
    } else {
      // Invalid input
      twiml.say({
        voice: 'alice',
        language: 'en-IN'
      }, 'Invalid input. Please try again.');
      twiml.redirect(`https://rail-madad-backend-p4vg.onrender.com/api/test/voice-handler/${complaintId}`);
    }

    res.type('text/xml');
    res.send(twiml.toString());
  } catch (error) {
    console.error("Voice Response Error:", error);
    
    // Even if there's an error, we need to return valid TwiML
    const twiml = new twilio.twiml.VoiceResponse();
    twiml.say({
      voice: 'alice',
      language: 'en-IN'
    }, 'We encountered an error. Please try again later.');
    twiml.hangup();
    
    res.type('text/xml');
    res.send(twiml.toString());
  }
};
// CHANGE 15: Improved Retry Call Function
async function retryCall(complaintId) {
  try {
    const complaint = await Complaint.findById(complaintId);
    console.log(complaint);
    // CHANGE 16: Multiple checks before retrying
    if (!complaint) {
      console.log(`❌ Complaint ${complaintId} not found for retry`);
      return;
    }
    
    if (complaint.acknowledgmentReceived) {
      console.log(`✅ Complaint ${complaintId} already acknowledged. Stopping retries.`);
      return;
    }
    
    if (complaint.callAttempts >= complaint.maxCallAttempts) {
      console.log(`❌ Max attempts reached for complaint ${complaintId}. No more retries.`);
      return;
    }

    const staff = await Staff.findOne({ 
      department: complaint.department,
      mobileNumber: complaint.assignedTo.phone 
    });
    
    if (!staff) {
      console.log(`❌ No staff found for retry of complaint ${complaintId}`);
      return;
    }

    // Increment call attempts
    const newAttemptCount = complaint.callAttempts + 1;
    
    // CHANGE 17: Build proper TwiML URL with all parameters
    let complaintDetails = '';
    if (complaint.details) complaintDetails += `Vivaran: ${complaint.details}. `;
    if (complaint.coachNo) complaintDetails += `Coach Number: ${complaint.coachNo}. `;
    
    const twimlUrl = `https://rail-madad-backend-p4vg.onrender.com/api/testvoice-handler/${complaintId}?category=${encodeURIComponent(complaint.category)}&subcategory=${encodeURIComponent(complaint.subCategory)}&details=${encodeURIComponent(complaintDetails)}&coach=${encodeURIComponent(complaint.coachNo || '')}&attempt=${newAttemptCount}`;
    
    const call = await twilioClient.calls.create({
      to: staff.mobileNumber,
      from: twilioNumber,
      url: twimlUrl,
      statusCallback: `https://rail-madad-backend-p4vg.onrender.com/api/test/call-status/${complaintId}`,
      statusCallbackEvent: ['answered', 'completed', 'no-answer', 'busy', 'failed'],
      timeout: 30,
      record: false
    });

    // CHANGE 18: Update complaint with retry information
    await Complaint.findByIdAndUpdate(complaintId, {
      $set: {
        callSid: call.sid,
        callAttempts: newAttemptCount,
        lastCallTime: new Date(),
        lastRetryReason: 'Previous call failed'
      },
      $push: {
        callHistory: {
          attempt: newAttemptCount,
          callSid: call.sid,
          status: 'initiated',
          timestamp: new Date(),
          isRetry: true
        }
      }
    });
    
    console.log(`✅ Retry call ${newAttemptCount} initiated for complaint ${complaintId}: ${call.sid}`);
  } catch (error) {
    console.error(`❌ Retry call error for complaint ${complaintId}:`, error);
    
    // CHANGE 19: If retry fails, try SMS immediately
    try {
      const complaint = await Complaint.findById(complaintId);
      if (complaint) {
        const staff = await Staff.findOne({ 
          department: complaint.department,
          mobileNumber: complaint.assignedTo?.phone 
        });
        
        if (staff) {
          await fallbackToSMS(complaintId, staff, "शिकायत", complaint.subCategory, complaint.details, complaint.coachNumber);
        }
      }
    } catch (smsError) {
      console.error(`❌ SMS fallback also failed for complaint ${complaintId}:`, smsError);
    }
  }
}

// CHANGE 20: Improved Fallback SMS Function
async function fallbackToSMS(complaintId, staff, category, subCategory, details, coachNumber) {
  try {
    // CHANGE 21: Include coach number in SMS
    let message = `नई शिकायत: ${subCategory} श्रेणी: ${category}.`;
    if (coachNumber) message += ` डिब्बा नंबर: ${coachNumber}.`;
    if (details) message += ` विवरण: ${details}`;
    message += ' कृपया तुरंत कार्रवाई करें। Reply ACK to confirm receipt.';
    
    const smsResponse = await twilioClient.messages.create({
      body: message,
      to: staff.mobileNumber,
      from: twilioNumber
    });
    
    console.log("✅ Fallback SMS sent:", smsResponse.sid);
    
    // CHANGE 22: Update complaint with SMS details
    await Complaint.findByIdAndUpdate(complaintId, {
      $set: {
        staffNotified: true,
        smsSid: smsResponse.sid,
        smsBackup: true,
        smsBackupTime: new Date(),
        notificationMethod: 'SMS_FALLBACK'
      }
    });
  } catch (smsError) {
    console.error("❌ SMS Error:", smsError);
    
    // CHANGE 23: Update complaint status if SMS also fails
    await Complaint.findByIdAndUpdate(complaintId, {
      $set: {
        status: "Notification Failed",
        notificationErrors: true,
        lastErrorTime: new Date()
      }
    });
  }
}

// CHANGE 24: Improved Background Job - Run every 2 minutes instead of 1 minute
setInterval(async () => {
  try {
    const fiveMinutesAgo = new Date(Date.now() - 5 * 60 * 1000);
    
    const unacknowledgedComplaints = await Complaint.find({
      createdAt: { $lte: fiveMinutesAgo },
      acknowledgmentReceived: false,
      status: { $nin: ['Resolved', 'Closed', 'Acknowledged'] },
      callAttempts: { $lt: 3 } // Only check complaints that haven't reached max attempts
    });

    console.log(`🔍 Background check: Found ${unacknowledgedComplaints.length} unacknowledged complaints`);

    for (const complaint of unacknowledgedComplaints) {
      console.log(`Checking unacknowledged complaint ${complaint._id} - Attempts: ${complaint.callAttempts}`);
      
      if (complaint.callAttempts < complaint.maxCallAttempts) {
        // Check if enough time has passed since last call
        const timeSinceLastCall = complaint.lastCallTime ? 
          (Date.now() - complaint.lastCallTime.getTime()) : Infinity;
          
        if (timeSinceLastCall > 300000) { // 5 minutes since last call
          console.log(`Initiating retry for complaint ${complaint._id}`);
          await retryCall(complaint._id);
        }
      } else if (!complaint.smsBackup) {
        // Send SMS if not already sent and calls failed
        const staff = await Staff.findOne({ 
          department: complaint.department,
          mobileNumber: complaint.assignedTo?.phone 
        });
        
        if (staff) {
          console.log(`Sending SMS fallback for complaint ${complaint._id}`);
          await fallbackToSMS(complaint._id, staff, "शिकायत", complaint.subCategory, complaint.details, complaint.coachNumber);
        }
      }
    }
  } catch (error) {
    console.error("Background check error:", error);
  }
}, 120000); // Check every 2 minutes instead of 1 minute
// 