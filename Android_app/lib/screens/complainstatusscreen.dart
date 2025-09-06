import 'dart:async';
import 'dart:convert';

import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;

class ComplaintStatusScreen extends StatefulWidget {
  final String category;
  final String subCategory;
  final String complaintId;

  const ComplaintStatusScreen({
    super.key,
    required this.category,
    required this.subCategory,
    required this.complaintId,
  });

  @override
  State<ComplaintStatusScreen> createState() => _ComplaintStatusScreenState();
}

class _ComplaintStatusScreenState extends State<ComplaintStatusScreen>
    with SingleTickerProviderStateMixin {
  late AnimationController _animationController;
  late Animation<double> _fadeAnimation;
  late Animation<Offset> _slideAnimation;

  int _secondsRemaining = 600; // 10 minutes
  bool _isResolved = false;
  bool _showEscalation = false;
  bool _isUpdating = false;

  Timer? _countdownTimer;
  Timer? _refreshTimer;

  @override
  void initState() {
    super.initState();

    _animationController = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 800),
    );
    _fadeAnimation = Tween<double>(begin: 0.0, end: 1.0).animate(
      CurvedAnimation(
        parent: _animationController,
        curve: const Interval(0.0, 0.5, curve: Curves.easeInOut),
      ),
    );
    _slideAnimation = Tween<Offset>(
      begin: const Offset(0, 0.5),
      end: Offset.zero,
    ).animate(
      CurvedAnimation(
        parent: _animationController,
        curve: const Interval(0.5, 1.0, curve: Curves.easeOut),
      ),
    );
    _animationController.forward();

    _startCountdown();
    _startAutoRefresh();
  }

  @override
  void dispose() {
    _animationController.dispose();
    _countdownTimer?.cancel();
    _refreshTimer?.cancel();
    super.dispose();
  }

  void _startCountdown() {
    _countdownTimer = Timer.periodic(const Duration(seconds: 1), (_) {
      if (!mounted) return;
      if (_isResolved || _secondsRemaining <= 0) {
        _countdownTimer?.cancel();
      } else {
        setState(() {
          _secondsRemaining--;
          if (_secondsRemaining <= 300) {
            _showEscalation = true;
          }
        });
      }
    });
  }

  void _startAutoRefresh() {
    _refreshTimer = Timer.periodic(const Duration(seconds: 30), (_) {
      if (_isResolved) {
        _refreshTimer?.cancel();
      } else {
        _fetchComplaintStatus();
      }
    });
    // initial fetch
    _fetchComplaintStatus();
  }

  Future<void> _fetchComplaintStatus() async {
    try {
      final resp = await http.get(
        Uri.parse('https://rail-madad-backend-p4vg.onrender.com/api/complaint/${widget.complaintId}'),
      );
      if (resp.statusCode == 200) {
        final data = jsonDecode(resp.body);
        final ack = data['acknowledgmentReceived'] as bool? ?? false;
        if (ack && !_isResolved) {
          setState(() {
            _isResolved = true;
          });
          _countdownTimer?.cancel();
          _refreshTimer?.cancel();
        }
      }
    } catch (_) {}
  }

  Future<void> _markAsResolved() async {
    setState(() => _isUpdating = true);
    try {
      final resp = await http.put(
        Uri.parse(
            'https://rail-madad-backend-p4vg.onrender.com/api/send/complainto/complaint/${widget.complaintId}/resolved'),
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode({'resolved': true}),
      );
      if (resp.statusCode == 200) {
        setState(() {
          _isResolved = true;
        });
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(
            content: Text('Complaint marked as resolved!'),
            backgroundColor: Colors.green,
          ),
        );
      } else {
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(
            content:
            Text('Failed to update complaint status. Please try again.'),
            backgroundColor: Colors.red,
          ),
        );
      }
    } catch (e) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Text('Error: ${e.toString()}'),
          backgroundColor: Colors.red,
        ),
      );
    } finally {
      setState(() => _isUpdating = false);
    }
  }

  void _escalateComplaint() {
    ScaffoldMessenger.of(context).showSnackBar(
      const SnackBar(
        content: Text('Complaint has been escalated to higher authorities!'),
        backgroundColor: Colors.orange,
      ),
    );
    // Add actual escalation API call here if needed
  }

  @override
  Widget build(BuildContext context) {
    final minutes = (_secondsRemaining ~/ 60).toString().padLeft(2, '0');
    final seconds = (_secondsRemaining % 60).toString().padLeft(2, '0');

    return Scaffold(
      appBar: AppBar(
        title: const Text('Complaint Status'),
        backgroundColor: const Color(0xFF1A237E),
      ),
      body: Container(
        padding: const EdgeInsets.all(16),
        decoration: BoxDecoration(
          gradient: LinearGradient(
            colors: [Colors.white, Colors.blue.shade50],
            begin: Alignment.topCenter,
            end: Alignment.bottomCenter,
          ),
        ),
        child: ListView(
          children: [
            FadeTransition(
              opacity: _fadeAnimation,
              child: SlideTransition(
                position: _slideAnimation,
                child: const Text(
                  'Complaint Status',
                  style: TextStyle(
                    fontSize: 28,
                    fontWeight: FontWeight.bold,
                    color: Color(0xFF1A237E),
                  ),
                ),
              ),
            ),
            const SizedBox(height: 20),
            FadeTransition(
              opacity: _fadeAnimation,
              child: SlideTransition(
                position: _slideAnimation,
                child: Card(
                  elevation: 6,
                  shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(20)),
                  shadowColor: const Color(0xFF1A237E).withOpacity(0.3),
                  child: Padding(
                    padding: const EdgeInsets.all(20),
                    child: Column(
                      children: [
                        Icon(
                          _isResolved
                              ? Icons.check_circle
                              : Icons.access_time,
                          size: 64,
                          color:
                          _isResolved ? Colors.green : Colors.blue,
                        ),
                        const SizedBox(height: 16),
                        Text(
                          _isResolved
                              ? 'Complaint Resolved Successfully!'
                              : 'Complaint Submitted Successfully!',
                          style: const TextStyle(
                            fontSize: 20,
                            fontWeight: FontWeight.bold,
                          ),
                          textAlign: TextAlign.center,
                        ),
                        const SizedBox(height: 12),
                        Text(
                          '${widget.category} - ${widget.subCategory}',
                          style: const TextStyle(
                            fontSize: 16,
                            color: Colors.grey,
                          ),
                          textAlign: TextAlign.center,
                        ),
                        const SizedBox(height: 8),
                        Text(
                          'Complaint ID: ${widget.complaintId}',
                          style: const TextStyle(
                            fontSize: 14,
                            color: Colors.grey,
                          ),
                          textAlign: TextAlign.center,
                        ),
                      ],
                    ),
                  ),
                ),
              ),
            ),
            if (!_isResolved) ...[
              const SizedBox(height: 28),
              FadeTransition(
                opacity: _fadeAnimation,
                child: SlideTransition(
                  position: _slideAnimation,
                  child: const Text(
                    'Resolution Timeline',
                    style: TextStyle(
                      fontSize: 22,
                      fontWeight: FontWeight.bold,
                      color: Color(0xFF1A237E),
                    ),
                  ),
                ),
              ),
              const SizedBox(height: 16),
              FadeTransition(
                opacity: _fadeAnimation,
                child: SlideTransition(
                  position: _slideAnimation,
                  child: Card(
                    elevation: 4,
                    shape: RoundedRectangleBorder(
                        borderRadius: BorderRadius.circular(20)),
                    child: Padding(
                      padding: const EdgeInsets.all(20),
                      child: Column(
                        children: [
                          const Text(
                            'Time remaining for resolution:',
                            style: TextStyle(fontSize: 16),
                          ),
                          const SizedBox(height: 12),
                          Text(
                            '$minutes:$seconds',
                            style: TextStyle(
                              fontSize: 36,
                              fontWeight: FontWeight.bold,
                              color: _secondsRemaining > 300
                                  ? Colors.green
                                  : Colors.orange,
                            ),
                          ),
                          const SizedBox(height: 16),
                          LinearProgressIndicator(
                            value: _secondsRemaining / 600,
                            backgroundColor: Colors.grey[300],
                            valueColor: AlwaysStoppedAnimation<Color>(
                              _secondsRemaining > 300
                                  ? Colors.green
                                  : Colors.orange,
                            ),
                            minHeight: 12,
                          ),
                        ],
                      ),
                    ),
                  ),
                ),
              ),
            ],
            const SizedBox(height: 28),
            if (_isResolved)
              FadeTransition(
                opacity: _fadeAnimation,
                child: SlideTransition(
                  position: _slideAnimation,
                  child: Card(
                    elevation: 4,
                    shape: RoundedRectangleBorder(
                        borderRadius: BorderRadius.circular(20)),
                    color: Colors.green,
                    child: const Padding(
                      padding: EdgeInsets.all(16),
                      child: Row(
                        children: [
                          Icon(Icons.check_circle, color: Colors.white),
                          SizedBox(width: 12),
                          Expanded(
                            child: Text(
                              'Complaint resolved successfully!',
                              style: TextStyle(
                                color: Colors.white,
                                fontWeight: FontWeight.bold,
                              ),
                            ),
                          ),
                        ],
                      ),
                    ),
                  ),
                ),
              )
            else if (_showEscalation)
              FadeTransition(
                opacity: _fadeAnimation,
                child: SlideTransition(
                  position: _slideAnimation,
                  child: Card(
                    elevation: 4,
                    shape: RoundedRectangleBorder(
                        borderRadius: BorderRadius.circular(20)),
                    color: Colors.orange,
                    child: Padding(
                      padding: const EdgeInsets.all(16),
                      child: Column(
                        children: [
                          const Row(
                            children: [
                              Icon(Icons.warning,
                                  color: Colors.white),
                              SizedBox(width: 12),
                              Expanded(
                                child: Text(
                                  'Complaint not resolved in time!',
                                  style: TextStyle(
                                    color: Colors.white,
                                    fontWeight: FontWeight.bold,
                                  ),
                                ),
                              ),
                            ],
                          ),
                          const SizedBox(height: 16),
                          SizedBox(
                            width: double.infinity,
                            child: ElevatedButton(
                              onPressed: _escalateComplaint,
                              style: ElevatedButton.styleFrom(
                                backgroundColor: Colors.white,
                                shape: RoundedRectangleBorder(
                                  borderRadius:
                                  BorderRadius.circular(12),
                                ),
                              ),
                              child: const Text(
                                'Escalate Complaint',
                                style: TextStyle(
                                  color: Colors.orange,
                                  fontWeight: FontWeight.bold,
                                ),
                              ),
                            ),
                          ),
                        ],
                      ),
                    ),
                  ),
                ),
              )
            else ...[
                const SizedBox(height: 28),
                FadeTransition(
                  opacity: _fadeAnimation,
                  child: SlideTransition(
                    position: _slideAnimation,
                    child: SizedBox(
                      width: double.infinity,
                      height: 50,
                      child: ElevatedButton(
                        onPressed:
                        _isUpdating ? null : _markAsResolved,
                        style: ElevatedButton.styleFrom(
                          backgroundColor: const Color(0xFF1A237E),
                          foregroundColor: Colors.white,
                          shape: RoundedRectangleBorder(
                            borderRadius:
                            BorderRadius.circular(12),
                          ),
                        ),
                        child: _isUpdating
                            ? const SizedBox(
                          width: 20,
                          height: 20,
                          child: CircularProgressIndicator(
                            color: Colors.white,
                            strokeWidth: 2,
                          ),
                        )
                            : const Text(
                          'Mark as Resolved',
                          style: TextStyle(
                            fontSize: 16,
                            fontWeight: FontWeight.bold,
                          ),
                        ),
                      ),
                    ),
                  ),
                ),
              ],
          ],
        ),
      ),
    );
  }
}
