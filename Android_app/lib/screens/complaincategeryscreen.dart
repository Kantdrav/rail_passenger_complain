import 'package:flutter/material.dart';
import 'package:flutter/scheduler.dart' show timeDilation;
import 'package:rail_complaint/screens/profilescreen.dart';

import 'homescreen.dart';

class ComplaintCategoriesScreen extends StatefulWidget {
  const ComplaintCategoriesScreen({super.key});

  @override
  State<ComplaintCategoriesScreen> createState() => _ComplaintCategoriesScreenState();
}

class _ComplaintCategoriesScreenState extends State<ComplaintCategoriesScreen> with SingleTickerProviderStateMixin {
  late AnimationController _animationController;
  late Animation<double> _fadeAnimation;
  late Animation<Offset> _slideAnimation;

  final List<ComplaintCategory> categories = [
    ComplaintCategory(
      name: 'Coach & Seat',
      icon: Icons.chair,
      subCategories: [
        'Seat broken',
        'Seat dirty',
        'Backrest not working',
        'Seat cushion missing',
      ],
    ),
    ComplaintCategory(
      name: 'Cleanliness',
      icon: Icons.clean_hands,
      subCategories: [
        'Toilet dirty',
        'Coach dirty',
        'Garbage not collected',
        'Bedding not clean',
      ],
    ),
    ComplaintCategory(
      name: 'Food & Catering',
      icon: Icons.restaurant,
      subCategories: [
        'Food quality poor',
        'Food not delivered',
        'Overcharged',
        'Wrong order',
      ],
    ),
    ComplaintCategory(
      name: 'AC & Temperature',
      icon: Icons.ac_unit,
      subCategories: [
        'AC not working',
        'Too cold',
        'Too hot',
        'Ventilation issue',
      ],
    ),
    ComplaintCategory(
      name: 'Electrical',
      icon: Icons.electrical_services,
      subCategories: [
        'Light not working',
        'Charging point not working',
        'Fan not working',
        'Switch not working',
      ],
    ),
    ComplaintCategory(
      name: 'Staff Behavior',
      icon: Icons.people,
      subCategories: [
        'Rude behavior',
        'Not helpful',
        'Not available',
        'Language issue',
      ],
    ),
    // New categories added
    ComplaintCategory(
      name: 'Luggage Issues',
      icon: Icons.luggage,
      subCategories: [
        'Lost luggage',
        'Damaged luggage',
        'Theft from luggage',
        'Luggage space unavailable',
      ],
    ),
    ComplaintCategory(
      name: 'Ticketing',
      icon: Icons.confirmation_number,
      subCategories: [
        'Overcharged for ticket',
        'Wrong ticket issued',
        'Refund not processed',
        'Online booking issue',
      ],
    ),
    ComplaintCategory(
      name: 'Safety & Security',
      icon: Icons.security,
      subCategories: [
        'Suspicious activity',
        'Theft incident',
        'Harassment',
        'Emergency help not available',
      ],
    ),
    ComplaintCategory(
      name: 'Punctuality',
      icon: Icons.schedule,
      subCategories: [
        'Train delayed',
        'Early departure',
        'No announcement for delay',
        'Connection missed due to delay',
      ],
    ),
    ComplaintCategory(
      name: 'Facilities',
      icon: Icons.escalator_warning,
      subCategories: [
        'Escalator not working',
        'Lift not working',
        'Waiting room issues',
        'Drinking water not available',
      ],
    ),
    ComplaintCategory(
      name: 'Accessibility',
      icon: Icons.accessible,
      subCategories: [
        'Wheelchair access issue',
        'No assistance for disabled',
        'Ramp not available',
        'Special needs not accommodated',
      ],
    ),
  ];

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
  }

  @override
  void dispose() {
    _animationController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Select Category'),
        backgroundColor: const Color(0xFF1A237E),
        foregroundColor: Colors.white,
        elevation: 4,
      ),
      body: Container(
        padding: const EdgeInsets.all(16.0),
        decoration: BoxDecoration(
          gradient: LinearGradient(
            begin: Alignment.topCenter,
            end: Alignment.bottomCenter,
            colors: [
              Colors.white,
              Colors.blue.shade50,
            ],
          ),
        ),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            FadeTransition(
              opacity: _fadeAnimation,
              child: SlideTransition(
                position: _slideAnimation,
                child: const Text(
                  'Choose a category for your complaint',
                  style: TextStyle(
                    fontSize: 20,
                    fontWeight: FontWeight.bold,
                    color: Color(0xFF1A237E),
                  ),
                ),
              ),
            ),
            const SizedBox(height: 20),
            Expanded(
              child: GridView.builder(
                gridDelegate: const SliverGridDelegateWithFixedCrossAxisCount(
                  crossAxisCount: 2,
                  crossAxisSpacing: 16,
                  mainAxisSpacing: 16,
                  childAspectRatio: 1.1,
                ),
                itemCount: categories.length,
                itemBuilder: (context, index) {
                  return FadeTransition(
                    opacity: _fadeAnimation,
                    child: SlideTransition(
                      position: _slideAnimation,
                      child: CategoryCard(
                        category: categories[index],
                        onTap: () {
                          Navigator.push(
                            context,
                            MaterialPageRoute(
                              builder: (context) => SubCategoryScreen(
                                category: categories[index],
                              ),
                            ),
                          );
                        },
                      ),
                    ),
                  );
                },
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class CategoryCard extends StatelessWidget {
  final ComplaintCategory category;
  final VoidCallback onTap;

  const CategoryCard({
    super.key,
    required this.category,
    required this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    return Card(
      elevation: 4,
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(16),
      ),
      shadowColor: const Color(0xFF1A237E).withOpacity(0.2),
      child: InkWell(
        onTap: onTap,
        borderRadius: BorderRadius.circular(16),
        child: Container(
          decoration: BoxDecoration(
            borderRadius: BorderRadius.circular(16),
            gradient: const LinearGradient(
              begin: Alignment.topLeft,
              end: Alignment.bottomRight,
              colors: [
                Color(0xFFE8EAF6),
                Colors.white,
              ],
            ),
          ),
          child: Padding(
            padding: const EdgeInsets.all(16.0),
            child: Column(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                Icon(
                  category.icon,
                  size: 36,
                  color: const Color(0xFF1A237E),
                ),
                const SizedBox(height: 12),
                Text(
                  category.name,
                  textAlign: TextAlign.center,
                  style: const TextStyle(
                    fontWeight: FontWeight.w600,
                    fontSize: 15,
                    color: Color(0xFF1A237E),
                  ),
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }
}

class ComplaintCategory {
  final String name;
  final IconData icon;
  final List<String> subCategories;

  ComplaintCategory({
    required this.name,
    required this.icon,
    required this.subCategories,
  });
}

class SubCategoryScreen extends StatefulWidget {
  final ComplaintCategory category;

  const SubCategoryScreen({super.key, required this.category});

  @override
  State<SubCategoryScreen> createState() => _SubCategoryScreenState();
}

class _SubCategoryScreenState extends State<SubCategoryScreen> with SingleTickerProviderStateMixin {
  late AnimationController _animationController;
  late Animation<double> _fadeAnimation;
  late Animation<Offset> _slideAnimation;
  String? _selectedSubCategory;

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
  }

  @override
  void dispose() {
    _animationController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text(widget.category.name),
        backgroundColor: const Color(0xFF1A237E),
        foregroundColor: Colors.white,
        elevation: 4,
      ),
      body: Container(
        padding: const EdgeInsets.all(16.0),
        decoration: BoxDecoration(
          gradient: LinearGradient(
            begin: Alignment.topCenter,
            end: Alignment.bottomCenter,
            colors: [
              Colors.white,
              Colors.blue.shade50,
            ],
          ),
        ),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            FadeTransition(
              opacity: _fadeAnimation,
              child: SlideTransition(
                position: _slideAnimation,
                child: const Text(
                  'Select specific issue',
                  style: TextStyle(
                    fontSize: 20,
                    fontWeight: FontWeight.bold,
                    color: Color(0xFF1A237E),
                  ),
                ),
              ),
            ),
            const SizedBox(height: 20),
            Expanded(
              child: ListView.builder(
                itemCount: widget.category.subCategories.length,
                itemBuilder: (context, index) {
                  final subCategory = widget.category.subCategories[index];
                  return FadeTransition(
                    opacity: _fadeAnimation,
                    child: SlideTransition(
                      position: _slideAnimation,
                      child: Card(
                        margin: const EdgeInsets.only(bottom: 16),
                        shape: RoundedRectangleBorder(
                          borderRadius: BorderRadius.circular(16),
                        ),
                        elevation: 3,
                        color: _selectedSubCategory == subCategory
                            ? const Color(0xFF1A237E).withOpacity(0.1)
                            : null,
                        child: ListTile(
                          leading: Icon(
                            widget.category.icon,
                            color: const Color(0xFF1A237E),
                          ),
                          title: Text(
                            subCategory,
                            style: const TextStyle(fontWeight: FontWeight.w500),
                          ),
                          trailing: _selectedSubCategory == subCategory
                              ? const Icon(Icons.check_circle, color: Color(0xFF1A237E))
                              : null,
                          onTap: () {
                            setState(() {
                              _selectedSubCategory = subCategory;
                            });

                            // Navigate to complaint details screen after a short delay
                            Future.delayed(const Duration(milliseconds: 300), () {
                              Navigator.push(
                                context,
                                MaterialPageRoute(
                                  builder: (context) => ComplaintDetailsScreen(
                                    category: widget.category.name,
                                    subCategory: subCategory,
                                  ),
                                ),
                              );
                            });
                          },
                        ),
                      ),
                    ),
                  );
                },
              ),
            ),
          ],
        ),
      ),
    );
  }
}
