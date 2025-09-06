import nodemailer from "nodemailer";
import jwt from "jsonwebtoken";
import express from "express";
import bcrypt from "bcryptjs";
import twilio from "twilio";
import Complaint from "../schemas/complaintschema.js";
import Staff from "../schemas/staffschema.js";

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

export const getAllcomplaint =  async (req, res) => {
  try {
    // Assuming you have user authentication and can get the user ID from the request
    // For now, we'll just get the most recent complaints
    const recentComplaints = await Complaint.find()
      .sort({ createdAt: -1 }) // Sort by most recent
      .limit(10) // Limit to 10 complaints
      .select('category subCategory status trainNo createdAt'); // Select only necessary fields
    
    res.status(200).json({
      success: true,
      complaints: recentComplaints
    });
  } catch (error) {
    console.error('Error fetching recent complaints:', error);
    res.status(500).json({
      success: false,
      message: 'Server error'
    });
  }
}
export const statusupdate = async (req, res) => {
    try {
        const { complaintId } = req.params;
        const { resolved } = req.body;

        if (typeof resolved !== "boolean") {
            return res.status(400).json({ success: false, message: "Resolved must be true or false" });
        }

        const complaint = await Complaint.findById(complaintId);
        if (!complaint) {
            return res.status(404).json({ success: false, message: "Complaint not found" });
        }

        // If marking as resolved, calculate resolution time
        if (resolved === true) {
            const createdAt = new Date(complaint.createdAt);
            const resolvedAt = new Date();
            const resolutionTimeMs = resolvedAt - createdAt;
            const resolutionTimeMinutes = Math.floor(resolutionTimeMs / (1000 * 60));
            
            // Add resolution time information to the complaint
            complaint.resolvedAt = resolvedAt;
            complaint.resolutionTimeMinutes = resolutionTimeMinutes;
            complaint.resolvedWithin5Min = resolutionTimeMinutes <= 5;
        } else {
            // If marking as not resolved, clear resolution data
            complaint.resolvedAt = null;
            complaint.resolutionTimeMinutes = null;
            complaint.resolvedWithin5Min = false;
        }

        complaint.resolved = resolved;
        await complaint.save();
        
        console.log(complaint);

        res.status(200).json({ 
            success: true, 
            complaint, 
            message: `Complaint marked as ${resolved ? "Resolved" : "Not Resolved"}`,
            resolutionTime: complaint.resolutionTimeMinutes,
            resolvedWithin5Min: complaint.resolvedWithin5Min
        });
    } catch (error) {
        console.error("Update Resolved Status Error:", error);
        res.status(500).json({ success: false, message: "Server Error" });
    }
}
export const submitComplaint = async (req, res) => {
  try {
    console.log("twilioClient status:", twilioClient ? "✅ INITIALIZED" : "❌ NOT INITIALIZED");

    const { category, subCategory, details, trainNo, coachNo, pnrNumber } = req.body;

    // Validate required fields
    if (!category || !subCategory || !trainNo) {
      return res.status(400).json({
        success: false,
        message: "Category, subcategory, and train number are required"
      });
    }

    // 1. Category to Hindi mapping for voice call
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

    // Department mapping remains same
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

    // 2. Save complaint
    const complaint = new Complaint({
      category,
      subCategory,
      details,
      trainNo,
      coachNo,
      pnrNumber,
      department,
      status: "Pending"
    });

    await complaint.save();

    // 3. Find staff
    let staff = await Staff.findOne({ department, trainNo });
    console.log(`Staff found for train ${trainNo}:`, staff);
    
    if (!staff) {
      staff = await Staff.findOne({ department });
      console.log(`No staff found for train ${trainNo}, using general department staff:`, staff);
    }

    if (staff && twilioClient) {
      // 4. Trigger automated call in Hindi
      try {
        const hindiCategory = categoryHindi[category] || category;
        
      const call = await twilioClient.calls.create({
  twiml: `<Response>
            <Say voice="alice" language="en-IN">
              Namaskar. Train number ${trainNo} ke liye nayi shikayat darj ki gayi hai.
              Category: ${hindiCategory}.
              Samasya: ${subCategory}.
              ${details ? `Vivaran: ${details}.` : ''}
              Kripaya turant aavashyak karyavahi karein.
              Dhanyawad.
            </Say>
         </Response>`,
  to: staff.mobileNumber,
  from: twilioNumber
});


        console.log("✅ Hindi call initiated:", call.sid);
        complaint.staffCalled = true;
        complaint.assignedTo = {
          name: staff.name,
          phone: staff.mobileNumber,
          department: staff.department
        };
        await complaint.save();

      } catch (err) {
        console.error("❌ Twilio Call Error:", err);

        // Fallback to Hindi SMS
        try {
          const message = await twilioClient.messages.create({
            body: `नई शिकायत: ${subCategory} ट्रेन ${trainNo} में। ${details || ''} कृपया तुरंत कार्रवाई करें।`,
            to: staff.mobileNumber,
            from: twilioNumber
          });

          console.log("✅ Hindi SMS sent:", message.sid);
          complaint.staffNotified = true;
          await complaint.save();
        } catch (smsError) {
          console.error("❌ SMS Error:", smsError);
        }
      }
    } else {
      console.log(`❌ No staff found for department: ${department} or Twilio not configured`);
      complaint.status = "Pending - No staff assigned";
      await complaint.save();
    }

    res.status(201).json({
      success: true,
      complaint,
      message: staff ? "Complaint submitted and staff notified" : "Complaint submitted but no staff assigned"
    });
  } catch (error) {
    console.error("Complaint Submit Error:", error);
    res.status(500).json({ success: false, message: "Server Error" });
  }
};
